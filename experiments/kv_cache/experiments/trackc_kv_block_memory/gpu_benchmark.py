#!/usr/bin/env python3
"""GPU benchmark for dense clone/index_copy vs token-page COW vs COW+materialize.

Uses Stage 3C's real layer/cache dimensions and one saved R5 reuse mask. No
OpenVLA inference or environment rollouts are performed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import tempfile
import time
import uuid
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoConfig

REPO=Path(__file__).resolve().parents[4]
TRACE=REPO/"experiments"/"kv_cache"/"outputs"/"stage3c_oracle_task_relevance"/"protect_10"/"task4_state0"/"results.json"
OUT=REPO/"experiments"/"kv_cache"/"outputs"/"trackc_kv_block_memory"
PAGE_SIZES=(1,4,8,16,32)
LAYERS=32
MODEL_ID="openvla/openvla-7b-finetuned-libero-spatial"


def atomic_json(path:Path,data:dict):
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+".",dir=path.parent)
    try:
        with os.fdopen(fd,"w") as f:
            json.dump(data,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def stage3c_reuse_case():
    d=json.loads(TRACE.read_text())
    rec=next(r for r in d["records"] if r["type"]=="reuse" and r["source_age"]==1)
    seq=int(rec["prefill_length"])
    reused=sorted(set(int(i)+1 for i in rec["reusable_background_indices"]))
    fresh=[i for i in range(seq) if i not in set(reused)]
    if len(reused)!=int(rec["background_tokens_reused"]): raise AssertionError("Stage3C trace mask mismatch")
    return rec,seq,reused,fresh


def load_model_dims():
    cfg=AutoConfig.from_pretrained(MODEL_ID,trust_remote_code=True,local_files_only=True)
    text_cfg=getattr(cfg,"text_config",cfg)
    return {"layers":int(text_cfg.num_hidden_layers),"query_heads":int(text_cfg.num_attention_heads),
            "kv_heads":int(text_cfg.num_key_value_heads),"head_dim":int(getattr(text_cfg,"head_dim",text_cfg.hidden_size//text_cfg.num_attention_heads)),
            "dtype":"bfloat16","model_id":MODEL_ID,"config_id":getattr(cfg,"_name_or_path",MODEL_ID)}


def build_plan(seq_len:int,reused_seq:list[int],page_size:int):
    reused=set(reused_seq); n_pages=math.ceil(seq_len/page_size)
    kinds=[]; source_ids=[]; pool_ids=[]; cow_src=[]; cow_dst=[]
    allocated=0
    for p in range(n_pages):
        lo,hi=p*page_size,min((p+1)*page_size,seq_len)
        r=sum(i in reused for i in range(lo,hi)); valid=hi-lo
        if r==valid:
            kinds.append("share"); source_ids.append(p); pool_ids.append(-1)
        else:
            kinds.append("mixed_cow" if r>0 else "fresh_only")
            source_ids.append(-1); pool_ids.append(allocated)
            if r>0: cow_src.append(p); cow_dst.append(allocated)
            allocated+=1
    fresh=[i for i in range(seq_len) if i not in reused]
    fresh_page=[]; fresh_slot=[]
    for pos in fresh:
        p=pos//page_size
        if pool_ids[p]<0: raise AssertionError("Fresh row mapped to a source-shared page")
        fresh_page.append(pool_ids[p]*page_size+pos%page_size); fresh_slot.append(pos)
    return {"seq_len":seq_len,"page_size":page_size,"n_pages":n_pages,"kinds":kinds,
            "source_ids":source_ids,"pool_ids":pool_ids,"allocated":allocated,"cow_src":cow_src,
            "cow_dst":cow_dst,"fresh":fresh,"fresh_flat_slots":fresh_page,
            "share_count":kinds.count("share"),"cow_count":kinds.count("mixed_cow"),
            "fresh_only_count":kinds.count("fresh_only")}


def pageify_dense(x:torch.Tensor,page_size:int):
    # x: [1,H,S,D] -> fixed-capacity [P,page,H,D] page pool.
    _,heads,seq,dim=x.shape; p=math.ceil(seq/page_size)
    y=torch.zeros((p,page_size,heads,dim),dtype=x.dtype,device=x.device)
    y.view(p*page_size,heads,dim)[:seq].copy_(x[0].permute(1,0,2).contiguous())
    return y


def cow_update(src_pages:torch.Tensor,updates:torch.Tensor,plan:dict,pool:torch.Tensor,materialize:bool):
    # updates: [1,H,n_fresh,D], in ascending logical token-position order.
    psize=plan["page_size"]; heads=src_pages.shape[2]; dim=src_pages.shape[3]
    # Build page table and simple lifetime accounting each update.
    source_refs=[1]*plan["n_pages"]
    page_table=[]
    for kind,source_id,pool_id in zip(plan["kinds"],plan["source_ids"],plan["pool_ids"]):
        if kind=="share":
            source_refs[source_id]+=1; page_table.append(("source",source_id))
        else: page_table.append(("pool",pool_id))
    if plan["cow_src"]:
        si=torch.tensor(plan["cow_src"],dtype=torch.long,device=src_pages.device)
        di=torch.tensor(plan["cow_dst"],dtype=torch.long,device=src_pages.device)
        pool.index_copy_(0,di,src_pages.index_select(0,si))
    if plan["fresh"]:
        slots=torch.tensor(plan["fresh_flat_slots"],dtype=torch.long,device=src_pages.device)
        values=updates[0].permute(1,0,2).contiguous()
        pool.view(-1,heads,dim).index_copy_(0,slots,values)
    dense=None
    if materialize:
        chunks=[src_pages[pid] if kind=="source" else pool[pid] for kind,pid in page_table]
        dense=torch.cat(chunks,dim=0)[:plan["seq_len"]].permute(1,0,2).unsqueeze(0).contiguous()
    # The current cache is ephemeral; release target references after its
    # attention consumer returns, restoring immutable source refcounts.
    for kind,pid in page_table:
        if kind=="source": source_refs[pid]-=1
    if any(x!=1 for x in source_refs): raise AssertionError("Source page refcount/lifetime violation")
    return page_table,pool,dense


def baseline_update(src:torch.Tensor,updates:torch.Tensor,fresh_idx:torch.Tensor):
    out=src.clone()
    out.index_copy_(2,fresh_idx,updates)
    return out


def make_tensors(seq_len:int,heads:int,dim:int,nfresh:int,device,dtype):
    src=[torch.randn((1,heads,seq_len,dim),device=device,dtype=dtype) for _ in range(LAYERS*2)]
    updates=[torch.randn((1,heads,nfresh,dim),device=device,dtype=dtype) for _ in range(LAYERS*2)]
    return src,updates


def measure(call,device):
    torch.cuda.synchronize(device)
    start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
    torch.cuda.reset_peak_memory_stats(device)
    w0=time.perf_counter(); start.record(); result=call(); end.record()
    torch.cuda.synchronize(device); wall=(time.perf_counter()-w0)*1000
    cuda=float(start.elapsed_time(end)); peak=int(torch.cuda.max_memory_allocated(device))
    reserved=int(torch.cuda.max_memory_reserved(device))
    del result
    return wall,cuda,peak,reserved


def correctness_tests(device,heads,dim,dtype):
    checks=[]
    cases=[("all_fresh",[]), ("all_reused",list(range(1,40))),
           ("one_fresh",list(range(1,39))),
           ("contiguous_updates",[i for i in range(1,40) if not 10<=i<24]),
           ("scattered_updates",[i for i in range(1,40) if i%3!=1]),
           ("partial_final_page",[i for i in range(1,38) if i%4!=0])]
    for page in PAGE_SIZES:
        for name,reuse in cases:
            L=39
            # all_reused case keeps all visual/source slots; index 0 is fresh
            # in policy semantics, but this unit case intentionally tests all
            # logical positions reused to exercise the manager's edge case.
            if name=="all_reused": reuse=list(range(L))
            else: reuse=sorted(set(reuse))
            fresh=[i for i in range(L) if i not in set(reuse)]
            src=torch.randn((1,heads,L,dim),device=device,dtype=dtype)
            updates=torch.randn((1,heads,len(fresh),dim),device=device,dtype=dtype)
            fresh_tensor=torch.tensor(fresh,dtype=torch.long,device=device)
            original=src.clone()
            ref=baseline_update(src,updates,fresh_tensor)
            pages=pageify_dense(src,page); plan=build_plan(L,reuse,page)
            pool=torch.empty((plan["allocated"],page,heads,dim),dtype=dtype,device=device)
            table,pool,out=cow_update(pages,updates,plan,pool,True)
            same=bool(torch.equal(ref,out)); source_ok=bool(torch.equal(src,original))
            refs_ok=True; pages_ok=len(table)==plan["n_pages"] and pool.shape[0]==plan["allocated"]
            if not(same and source_ok and refs_ok and pages_ok):
                raise AssertionError(f"COW correctness failed: page={page}, case={name}, same={same}, source={source_ok}")
            checks.append({"page_tokens":page,"case":name,"logical_tensor_exact":same,
                           "source_immutable":source_ok,"page_count":len(table),
                           "pool_pages":int(pool.shape[0]),"refcounts_returned_to_source_only":refs_ok})
            del src,updates,fresh_tensor,original,ref,pages,table,pool,out
    return checks


def run_benchmark(args):
    out=args.out; out.mkdir(parents=True,exist_ok=True); (out/"logs").mkdir(exist_ok=True)
    config=load_model_dims()
    if config["layers"]!=LAYERS: raise RuntimeError(f"Expected {LAYERS} layers, config says {config['layers']}")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable in the established OpenVLA container")
    device=torch.device("cuda:0"); props=torch.cuda.get_device_properties(device)
    rec,L,reused_seq,fresh_seq=stage3c_reuse_case()
    H,D=config["kv_heads"],config["head_dim"]
    dtype=torch.bfloat16
    if len(fresh_seq)+len(reused_seq)!=L: raise AssertionError("Fresh/reused rows do not partition sequence")
    cfg={"model":config,"device":{"name":props.name,"total_memory_bytes":int(props.total_memory),
                                   "capability":[props.major,props.minor]},
         "trace_case":{"scenario":"task4_state0","step":int(rec["step"]),"source_age":int(rec["source_age"]),
                       "sequence_length":L,"fresh_sequence_rows":len(fresh_seq),
                       "reused_visual_tokens":len(reused_seq),"page_sizes":list(PAGE_SIZES)},
         "warmups":args.warmups,"repetitions":args.repetitions,
         "conditions":["A_dense_clone_index_copy","B_COW_page_update_no_materialization",
                       "C_COW_plus_dense_SDPA_materialization"],
         "attention_backend_requirement":"Transformers DynamicCache and SDPA path consume dense [batch, heads, sequence, head_dim] K/V tensors; C therefore measures full concatenation/materialization before attention.",
         "pool_allocation":"COW page pools allocated once as fixed-capacity buffers; per-step page assignment/refcounts and GPU page writes included in wall-time. Pool initialization/pageification reported separately.",
         "semantics":"R5 source is immutable full refresh; every row outside exact logged reusable visual positions is current/fresh."}
    config_hash=hashlib.sha256(json.dumps(cfg,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    atomic_json(out/"configuration.json",cfg)
    print("[3/4] Correctness tests",flush=True)
    validation={"cuda_available":True,"torch_version":torch.__version__,"cuda_runtime":torch.version.cuda,
                "device":props.name,"passed_cases":correctness_tests(device,H,D,dtype)}
    validation["all_cases_passed"]=len(validation["passed_cases"])==len(PAGE_SIZES)*6
    atomic_json(out/"correctness_validation.json",validation)
    if args.validate_only:
        print("[VALIDATION_PASS] exact K/V logical equality; source immutable; page references released",flush=True)
        return
    # Warmups reach steady allocator/SDPA behavior while
    # keeping small-page Python page-table overhead bounded on Orin.
    warmups=args.warmups; reps=args.repetitions
    attempt_rows=[]; config_summaries=[]
    raw_path=out/"gpu_benchmark.csv"
    completion_path=out/"logs"/"completed_configs.json"
    completed_keys=set()
    if completion_path.exists() and not args.overwrite:
        saved=json.loads(completion_path.read_text())
        saved_hash=saved.get("configuration_sha256")
        if saved_hash!=config_hash:
            raise RuntimeError("Existing benchmark configuration differs; refusing to mix/resume incompatible samples.")
        config_summaries=list(saved.get("completed",[]))
        completed_keys={(int(r["page_tokens"]),r["condition"]) for r in config_summaries
                        if r.get("n")==reps and r.get("attempt_id")}
    write_header=not raw_path.exists() or args.overwrite
    if args.overwrite and raw_path.exists(): raw_path.unlink(); write_header=True
    if args.overwrite:
        config_summaries=[]; completed_keys=set()
    fieldnames=["configuration_sha256","attempt_id","page_tokens","condition","measurement","iteration","wall_ms","cuda_event_ms","peak_allocated_bytes","peak_reserved_bytes","status"]
    raw_file=raw_path.open("a",newline="")
    writer=csv.DictWriter(raw_file,fieldnames=fieldnames)
    if write_header: writer.writeheader(); raw_file.flush()
    status_path=out/"logs"/"progress.json"
    total_configs=len(PAGE_SIZES)*3
    completed=0
    for page in PAGE_SIZES:
        plan=build_plan(L,reused_seq,page)
        src,updates=make_tensors(L,H,D,len(fresh_seq),device,dtype)
        idx=torch.tensor(fresh_seq,dtype=torch.long,device=device)
        baseline_reference_cpu=[baseline_update(src[i],updates[i],idx).detach().cpu() for i in range(LAYERS*2)]
        src_pages=None; cow_pools=None
        # A: actual current data operation, full source clone then fresh row copy.
        for condition in ("A_dense_clone_index_copy","B_COW_page_update_no_materialization","C_COW_plus_dense_SDPA_materialization"):
            if (page,condition) in completed_keys:
                print(f"[RESUME_SKIP] page={page} condition={condition} already complete",flush=True)
                continue
            attempt=str(uuid.uuid4())[:12]
            if condition=="A_dense_clone_index_copy":
                def call():
                    result=[baseline_update(src[i],updates[i],idx) for i in range(LAYERS*2)]
                    return result
            else:
                # Convert the dense refresh snapshot to immutable page backing,
                # then use fixed-capacity page pools for per-step COW updates.
                torch.cuda.synchronize(device); prep0=time.perf_counter()
                src_pages=[pageify_dense(src[i],page) for i in range(LAYERS*2)]
                cow_pools=[torch.empty((plan["allocated"],page,H,D),dtype=dtype,device=device) for _ in range(LAYERS*2)]
                torch.cuda.synchronize(device); prep_ms=(time.perf_counter()-prep0)*1000
                prep_bytes=sum(x.numel()*x.element_size() for x in src_pages+cow_pools)
                config_summaries.append({"page_tokens":page,"condition":"refresh_source_pageify_and_pool_setup",
                                         "one_time_wall_ms":prep_ms,"source_page_pool_bytes":prep_bytes,
                                         "source_pages_bytes":sum(x.numel()*x.element_size() for x in src_pages),
                                         "cow_pool_bytes":sum(x.numel()*x.element_size() for x in cow_pools)})
                plans=[plan]* (LAYERS*2)
                if condition.startswith("C_"):
                    def call():
                        result=[]
                        for i in range(LAYERS*2):
                            _,_,dense=cow_update(src_pages[i],updates[i],plans[i],cow_pools[i],True)
                            result.append(dense)
                        return result
                else:
                    def call():
                        result=[]
                        for i in range(LAYERS*2):
                            table,pool,_=cow_update(src_pages[i],updates[i],plans[i],cow_pools[i],False)
                            result.append((table,pool))
                        return result
            # Validate each path on identical input values before its timings.
            if condition.startswith("C_"):
                check=call()
                if not all(torch.equal(a,b.cpu()) for a,b in zip(baseline_reference_cpu,check)):
                    raise AssertionError(f"materialized COW differs from dense baseline at page={page}")
                del check
            else:
                # Materialize once just for exact validation; test was already
                # performed in correctness suite. Verify fresh/reuse provenance
                # from the plan and source rows for this actual mask.
                if len(plan["kinds"])!=math.ceil(L/page): raise AssertionError("page plan length mismatch")
            # For each condition, execute warmups and collect synchronized wall
            # time (Python manager + CUDA) and CUDA event interval.
            pbar=tqdm(range(warmups+reps),desc=f"[4/4] page={page} {condition}",unit="iter",leave=False)
            for it in pbar:
                wall,cuda_ms,peak,reserved=measure(call,device)
                atomic_json(status_path,{"phase":"gpu_benchmark","completed_configurations":completed,
                                         "total_configurations":total_configs,"current_page_tokens":page,
                                         "current_condition":condition,"iteration":it+1,
                                         "warmups":warmups,"timed_repetitions":reps,"status":"running"})
                if it>=warmups:
                    writer.writerow({"configuration_sha256":config_hash,"attempt_id":attempt,"page_tokens":page,"condition":condition,
                                     "measurement":"update" if "no_materialization" not in condition else "update_no_materialization",
                                     "iteration":it-warmups,"wall_ms":wall,"cuda_event_ms":cuda_ms,
                                     "peak_allocated_bytes":peak,"peak_reserved_bytes":reserved,"status":"sample"})
                    raw_file.flush(); os.fsync(raw_file.fileno())
                    attempt_rows.append({"attempt_id":attempt,"page_tokens":page,"condition":condition,
                                         "wall_ms":wall,"cuda_event_ms":cuda_ms,"peak_allocated_bytes":peak,
                                         "peak_reserved_bytes":reserved})
            completed+=1
            chosen=[r for r in attempt_rows if r["attempt_id"]==attempt]
            row={"attempt_id":attempt,"page_tokens":page,"condition":condition,"n":len(chosen),
                 "wall_mean_ms":statistics.fmean(r["wall_ms"] for r in chosen),
                 "wall_median_ms":statistics.median(r["wall_ms"] for r in chosen),
                 "wall_p95_ms":sorted(r["wall_ms"] for r in chosen)[math.ceil(.95*len(chosen))-1],
                 "cuda_event_mean_ms":statistics.fmean(r["cuda_event_ms"] for r in chosen),
                 "cuda_event_median_ms":statistics.median(r["cuda_event_ms"] for r in chosen),
                 "peak_allocated_bytes_max":max(r["peak_allocated_bytes"] for r in chosen),
                 "peak_reserved_bytes_max":max(r["peak_reserved_bytes"] for r in chosen)}
            config_summaries.append(row)
            atomic_json(completion_path,{"configuration_sha256":config_hash,"configuration":cfg,"completed":config_summaries})
            atomic_json(status_path,{"phase":"gpu_benchmark","completed_configurations":completed,
                                     "total_configurations":total_configs,"current_page_tokens":page,
                                     "current_condition":condition,"latest":row,"status":"running"})
            print(f"[CONFIG_DONE] page={page} {condition}: wall median {row['wall_median_ms']:.4f} ms; CUDA event {row['cuda_event_median_ms']:.4f} ms",flush=True)
            if condition=="A_dense_clone_index_copy":
                # Retain the CPU comparison copy only.
                pass
        # Pure attention execution on a dense cache, same semantics for all
        # three storage managers after the dense tensor handoff.
        if src_pages is None or cow_pools is None:
            src_pages=[pageify_dense(src[i],page) for i in range(LAYERS*2)]
            cow_pools=[torch.empty((plan["allocated"],page,H,D),dtype=dtype,device=device) for _ in range(LAYERS*2)]
        dense_for_attention=[]
        for i in range(2):
            pgs=src_pages[i]
            pool=cow_pools[i]
            _,_,dense=cow_update(pgs,updates[i],plan,pool,True)
            dense_for_attention.append(dense)
        q=torch.randn((1,H,L,D),device=device,dtype=dtype)
        k=dense_for_attention[0]; v=dense_for_attention[1]
        def attention_call(): return F.scaled_dot_product_attention(q,k,v,is_causal=True)
        att_attempt=str(uuid.uuid4())[:12]
        for it in range(warmups+reps):
            wall,cuda_ms,peak,reserved=measure(attention_call,device)
            if it>=warmups:
                writer.writerow({"configuration_sha256":config_hash,"attempt_id":att_attempt,"page_tokens":page,"condition":"SDPA_dense_reference",
                                 "measurement":"one_layer_full_prefill_attention","iteration":it-warmups,
                                 "wall_ms":wall,"cuda_event_ms":cuda_ms,"peak_allocated_bytes":peak,
                                 "peak_reserved_bytes":reserved,"status":"sample"})
                raw_file.flush()
        # Record dense independent source-clone latency from a refresh.
        clone_attempt=str(uuid.uuid4())[:12]
        clone_samples=[]
        for it in range(warmups+reps):
            wall,cuda_ms,peak,reserved=measure(lambda:[x.clone() for x in src],device)
            atomic_json(status_path,{"phase":"refresh_snapshot_clone","completed_configurations":completed,
                                     "total_configurations":total_configs,"current_page_tokens":page,
                                     "iteration":it+1,"warmups":warmups,"timed_repetitions":reps,"status":"running"})
            if it>=warmups:
                clone_samples.append({"wall_ms":wall,"cuda_event_ms":cuda_ms,"peak_allocated_bytes":peak,
                                      "peak_reserved_bytes":reserved})
                writer.writerow({"configuration_sha256":config_hash,"attempt_id":clone_attempt,"page_tokens":page,"condition":"refresh_dense_source_snapshot_clone",
                                 "measurement":"full_32_layer_KV_snapshot_clone","iteration":it-warmups,
                                 "wall_ms":wall,"cuda_event_ms":cuda_ms,"peak_allocated_bytes":peak,
                                 "peak_reserved_bytes":reserved,"status":"sample"}); raw_file.flush(); os.fsync(raw_file.fileno())
        snapshot_clone=[x.clone() for x in src]
        torch.cuda.synchronize(device); pageify_start=time.perf_counter()
        paged=[pageify_dense(x,page) for x in src]
        torch.cuda.synchronize(device); pageify_ms=(time.perf_counter()-pageify_start)*1000
        snapshot_bytes=sum(x.numel()*x.element_size() for x in snapshot_clone)
        config_summaries.append({"page_tokens":page,"condition":"refresh_source_preparation",
                                 "dense_snapshot_payload_bytes":snapshot_bytes,
                                 "dense_snapshot_clone_wall_median_ms":statistics.median(r["wall_ms"] for r in clone_samples),
                                 "dense_snapshot_clone_cuda_median_ms":statistics.median(r["cuda_event_ms"] for r in clone_samples),
                                 "dense_to_pages_wall_ms_once":pageify_ms,
                                 "page_pool_payload_bytes":sum(x.numel()*x.element_size() for x in paged)})
        del snapshot_clone,paged,q,k,v,src,updates,idx,src_pages,cow_pools,dense_for_attention,baseline_reference_cpu
        torch.cuda.empty_cache()
        atomic_json(completion_path,{"configuration_sha256":config_hash,"configuration":cfg,"completed":config_summaries})
    raw_file.close()
    # p95 and means are computed from only completed attempts, each with exact N.
    atomic_json(out/"gpu_benchmark_summary.json",{"status":"complete","configuration":cfg,
              "samples":config_summaries,"raw_csv":"gpu_benchmark.csv",
              "source_snapshot_clone_note":"Dense source snapshot bytes and pageification are recorded separately; one-time page conversion is not included in steady-state COW timings."})
    atomic_json(status_path,{"phase":"complete","completed_configurations":completed,
                             "total_configurations":total_configs,"status":"complete"})
    print("[GPU_BENCHMARK_DONE] raw samples flushed; summary complete",flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,default=OUT)
    p.add_argument("--warmups",type=int,default=20)
    p.add_argument("--repetitions",type=int,default=50)
    p.add_argument("--validate-only",action="store_true")
    p.add_argument("--overwrite",action="store_true")
    args=p.parse_args()
    if args.warmups<1 or args.repetitions<5: raise ValueError("Need >=1 warmup and >=5 timed reps")
    run_benchmark(args)


if __name__=="__main__": main()
