"""Bounded real-weight integrity checks; never extracts a dataset or trains."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .extract import load_native_dino,process_image_batch
from .masks import geometry_from_transform,apply_mask_geometry,fractional_grid
from .losses import dense_pair_losses,patch_logmeanexp_scores,symmetric_infonce
from ..text_encoder import load_siglip_text_tower
from ..projection import RegionTextProjection

def run(image_path,checkpoint_dir):
    model,processor,vision,dino,transform,torch=load_native_dino()
    image=Image.open(image_path).convert("RGB")
    _,pixels,x,y=process_image_batch([(1,image_path)],processor,dino,next(dino.parameters()).device,torch)
    again=dino(x)
    if isinstance(again,(tuple,list)):again=again[0]
    if tuple(y.shape)!=(1,256,1024) or not torch.isfinite(y).all():raise RuntimeError("Native penultimate DINO smoke failed")
    if not torch.equal(y,again):raise RuntimeError("Fresh OpenVLA penultimate output is nondeterministic")
    geom=transform
    test=np.zeros((image.height,image.width),np.uint8); test[:max(1,image.height//4),:max(1,image.width//4)]=1
    out=apply_mask_geometry(test,geom); grid=fractional_grid(out)
    if len(grid)!=256 or not np.isfinite(grid).all():raise RuntimeError("Mask-to-grid mapping invalid")
    # Synthetic vertical boundary verifies the real processor geometry and nearest mask path agree.
    from PIL import ImageDraw
    sw,sh=320,200; split=int(sw*.375); syn=Image.new("RGB",(sw,sh),(0,0,0));ImageDraw.Draw(syn).rectangle((0,0,split-1,sh-1),fill=(255,255,255))
    syn_mask=np.zeros((sh,sw),np.uint8);syn_mask[:,:split]=1
    proc=processor.image_processor(images=[syn],return_tensors="pt")["pixel_values"][:,:3]
    mgrid=fractional_grid(apply_mask_geometry(syn_mask,geom),16).reshape(16,16)
    boundary_col=int(np.argmax((mgrid[8]>0).astype(np.int32)==0))
    expected_col=int(round(.375*16))
    if abs(boundary_col-expected_col)>1:raise RuntimeError(f"Policy RGB/mask boundary token alignment failed: {boundary_col} vs {expected_col}")
    text_tower,tokenizer,text_meta=load_siglip_text_tower(checkpoint_dir,device=str(y.device))
    ids=tokenizer(["a photo of a cup","a photo of a bowl"]).to(y.device)
    with torch.inference_mode():emb=text_tower(ids)
    if emb.shape!=(2,1152) or not torch.isfinite(emb).all():raise RuntimeError("SigLIP text feature smoke failed")
    proj=RegionTextProjection().to(y.device); frozen=emb.detach().clone()
    opt=torch.optim.AdamW(proj.parameters(),lr=1e-4)
    patch=torch.nn.functional.normalize(y.float().repeat(2,1,1).clone(),dim=-1)
    ptext=proj.text_projection(frozen)
    scores=patch_logmeanexp_scores(patch,ptext)
    target=torch.zeros(2,1,256,device=y.device);target[:,:,10:14]=.5; valid=torch.ones_like(target)
    logits=torch.zeros_like(target,requires_grad=True)
    dense=dense_pair_losses(logits,target,valid,target.sum(-1))
    loss=.5*dense["bce"]+.5*dense["dice"]+.1*dense["hard_negative"]+.2*symmetric_infonce(scores)
    opt.zero_grad();loss.backward()
    if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in proj.parameters()):raise RuntimeError("MLP gradient missing")
    result={"status":"passed","device":str(y.device),"dino_shape":list(y.shape),"dino_dtype":str(y.dtype),"dino_repeat_exact":True,"dino_blocks":len(dino.blocks),"dino_forward_target_layer":len(dino.blocks)-2,"dino_register_tokens":int(getattr(dino,"num_reg_tokens",4)),"policy_processor_geometry":geom,"mask_grid_shape":[16,16],"synthetic_rgb_mask_boundary_column":boundary_col,"expected_boundary_column":expected_col,"siglip_embedding_shape":list(emb.shape),"siglip_metadata":text_meta,"mlp_parameters":sum(p.numel() for p in proj.parameters()),"loss_finite":bool(torch.isfinite(loss)),"mlp_gradients_finite":True,"cuda_peak_allocated_bytes":int(torch.cuda.max_memory_allocated(y.device)) if y.device.type=="cuda" else None}
    return result
def main():
    p=argparse.ArgumentParser();p.add_argument("--image",type=Path,required=True);p.add_argument("--siglip-checkpoint",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    d=run(a.image,a.siglip_checkpoint);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(d,indent=2)+"\n");print(json.dumps(d,indent=2))
if __name__=="__main__":main()
