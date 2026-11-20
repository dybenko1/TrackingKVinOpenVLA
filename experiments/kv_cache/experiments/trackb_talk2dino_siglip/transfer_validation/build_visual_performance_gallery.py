#!/usr/bin/env python3
"""Render a visual audit from E2's saved scores, masks, and paired HDF5 RGBs.

No model is loaded. E2 simulator-state replay is not repeated; this only reads
recorded camera observations and the immutable, reconciled score cache.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math
from pathlib import Path
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image

ROLES=('manipulated_objects','destination_objects','robot_arm','robot_gripper','task_relevant_fixtures','robot_mount')
ROLE_COLORS={'manipulated_objects':'#00d66b','destination_objects':'#00a5ff','robot_arm':'#ffb000','robot_gripper':'#ff4fa3','task_relevant_fixtures':'#b56bff','robot_mount':'#fff000'}
THRESHOLDS={'V1':.55,'V2@0.20':.20,'V2@0.08':.08}

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def mask(probs,t): return np.max(probs.astype(np.float32),axis=0)>=t
def met(pred,gt):
 g=gt>0
 tp=int((pred&g).sum()); fp=int((pred&~g).sum()); fn=int((~pred&g).sum()); tn=256-tp-fp-fn
 return dict(tp=tp,fp=fp,fn=fn,tn=tn,recall=tp/(tp+fn) if tp+fn else 0.,false_background=fn/(tp+fn) if tp+fn else 0.,precision=tp/(tp+fp) if tp+fp else 0.,iou=tp/(tp+fp+fn) if tp+fp+fn else 0.,reusable=int((~pred).sum()))

def load_rows(out, manifest):
 rows=[]
 for npz in sorted((out/'role_reconciled_scoring_cache').rglob('*.npz')):
  z=np.load(npz,allow_pickle=False); md=json.loads(str(z['metadata'].item()))
  task=next(t for t in manifest['tasks'] if t['task_id']==md['task_id'])
  roleidx={r:i for i,r in enumerate(md['role_keys'])}
  roles={r:z['role_gt_occupancy'][:,roleidx[r]].astype(np.float32) if r in roleidx else np.zeros((32,256),np.float32) for r in ROLES}
  gt=np.maximum.reduce(list(roles.values()))
  hdf=Path(task['hdf5_path'].replace('/opt/openvla/','/opt/openvla/'))
  rows.append(dict(z=z,md=md,task=task,roles=roles,gt=gt,hdf=hdf,cache=npz))
 return rows

def extract_rgb(row, idx):
 with h5py.File(row['hdf'],'r') as f:
  key=f"data/{row['md']['demo']}/obs/agentview_rgb"
  arr=f[key][int(idx)]
 if arr.ndim==3 and arr.shape[0] in (3,4): arr=np.moveaxis(arr,0,-1)
 arr=np.asarray(arr)
 if arr.shape[-1]==4: arr=arr[...,:3]
 return arr.astype(np.uint8)

def overlay(ax, rgb, arr, cmap, alpha=.62):
 ax.imshow(rgb)
 ax.imshow(arr.reshape(16,16),cmap=cmap,alpha=alpha,interpolation='nearest',vmin=0,vmax=1,extent=(0,rgb.shape[1],rgb.shape[0],0))
 ax.set_xticks(np.arange(0,rgb.shape[1]+1,rgb.shape[1]/16),minor=True); ax.set_yticks(np.arange(0,rgb.shape[0]+1,rgb.shape[0]/16),minor=True)
 ax.grid(which='minor',color='white',lw=.35,alpha=.65); ax.set_xticks([]);ax.set_yticks([])

def panel(row, ti, dst, title=None):
 z,md=row['z'],row['md']; gt=row['gt'][ti]>0
 p1=z['v1_probabilities'][ti]; p2=z['v2_probabilities'][ti]
 preds={'V1':mask(p1,.55),'V2@0.20':mask(p2,.20),'V2@0.08':mask(p2,.08)}
 rgb=extract_rgb(row,int(z['state_indices'][ti]))
 fig,axs=plt.subplots(2,4,figsize=(19,10),constrained_layout=True)
 fig.suptitle(title or f"{md['split']} · {md['task_id']} · {md['demo']} · frame/state {int(z['state_indices'][ti])}\n{row['task']['instruction']}",fontsize=12)
 axs=axs.ravel(); axs[0].imshow(rgb); axs[0].set_title('Recorded RGB (paired HDF5)')
 # Role color target, deterministic precedence by requested role order.
 rgb_role=np.zeros((16,16,4),np.float32)
 for role in ROLES:
  occ=row['roles'][role][ti].reshape(16,16)>0
  if occ.any():
   c=ROLE_COLORS[role]; rgb_role[occ]=(*plt.matplotlib.colors.to_rgb(c),.75)
 axs[1].imshow(rgb);axs[1].imshow(rgb_role,interpolation='nearest',extent=(0,rgb.shape[1],rgb.shape[0],0));axs[1].set_title('Ground truth by role')
 for a,key in zip(axs[2:5],('V1','V2@0.20','V2@0.08')):
  overlay(a,rgb,preds[key],'Greens',.45);a.set_title(f"{key} predicted fresh · {int(preds[key].sum())}/256")
 # confusion colors: TP green, FN red, FP orange, TN dark transparent
 cm=np.zeros((16,16,4),np.float32); pr=preds['V2@0.20']; cal=preds['V2@0.08'];
 for condition,sel,color in [('TP',pr & gt,'#11c76f'),('FN',~pr & gt,'#ee3344'),('FP',pr&~gt,'#ff9900'),('TN',~pr&~gt,'#222222')]:
  rgba=plt.matplotlib.colors.to_rgba(color, .78 if condition!='TN' else .18); cm[sel.reshape(16,16)]=rgba
 a=axs[5];a.imshow(rgb);a.imshow(cm,interpolation='nearest',extent=(0,rgb.shape[1],rgb.shape[0],0));a.set_title('V2@0.20 errors: TP green / FN red / FP orange')
 # Difference map: changed cells useful correction vs unnecessary new protection; remaining FN
 diff=np.zeros((16,16,4),np.float32); switched=cal&~pr; useful=switched & gt; bad=switched&~gt; remaining=~cal & gt
 diff[useful.reshape(16,16)]=(*plt.matplotlib.colors.to_rgb('#00ff66'),.9)
 diff[bad.reshape(16,16)]=(*plt.matplotlib.colors.to_rgb('#ff9900'),.9)
 diff[remaining.reshape(16,16)]=(*plt.matplotlib.colors.to_rgb('#ed3344'),.8)
 a=axs[6];a.imshow(rgb);a.imshow(diff,interpolation='nearest',extent=(0,rgb.shape[1],rgb.shape[0],0));a.set_title(f"0.20→0.08: corrected {int(useful.sum())} · extra bg {int(bad.sum())} · FN left {int(remaining.sum())}")
 a=axs[7];
 probs=np.max(p2.astype(np.float32),axis=0).reshape(16,16); a.imshow(probs,cmap='viridis',vmin=0,vmax=1);a.contour(probs,[.08,.20],colors=['cyan','white'],linewidths=1);a.set_title('V2 max prompt probability; .08 cyan / .20 white');a.set_xticks(np.arange(-.5,16,1));a.set_yticks(np.arange(-.5,16,1));a.grid(color='white',lw=.25,alpha=.3)
 for ax in axs: ax.axis('off')
 # Actual token-count metrics; GT positive means fractional occupancy > 0.
 lines=[]
 for k,pred in preds.items():
  m=met(pred,gt);lines.append(f"{k}: TP/FP/FN/TN {m['tp']}/{m['fp']}/{m['fn']}/{m['tn']}  R {m['recall']:.2f} FNR {m['false_background']:.2f} P {m['precision']:.2f} IoU {m['iou']:.2f} reuse {m['reusable']}/256")
 fig.text(.02,.005,'Binary token metrics (GT occupancy > 0): '+' | '.join(lines),fontsize=8,ha='left')
 dst.parent.mkdir(parents=True,exist_ok=True);fig.savefig(dst,dpi=150,bbox_inches='tight');plt.close(fig)
 return preds,gt

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--e2-dir',type=Path,required=True);ap.add_argument('--dataset-root',type=Path,required=True);a=ap.parse_args()
 out=a.e2_dir; root=a.dataset_root; gallery=out/'visual_performance_gallery';
 for d in ('gallery','temporal','plots'): (gallery/d).mkdir(parents=True,exist_ok=True)
 manifest=json.loads((out/'libero_task_manifest.json').read_text());rows=load_rows(out,manifest)
 verified=set()
 for r in rows:
  r['hdf']=root/r['task']['suite']/r['task']['file_name']
  if not r['hdf'].exists(): raise FileNotFoundError(r['hdf'])
  # HDF5 content integrity checked once per source file against the frozen manifest.
  if r['hdf'] not in verified:
   if sha(r['hdf'])!=r['task']['hdf5_sha256']: raise RuntimeError('HDF5 SHA mismatch: '+str(r['hdf']))
   verified.add(r['hdf'])
 frames=[]
 for r in rows:
  for i in range(32):
   p1=mask(r['z']['v1_probabilities'][i],.55);p20=mask(r['z']['v2_probabilities'][i],.20);p08=mask(r['z']['v2_probabilities'][i],.08);gt=r['gt'][i]>0
   m1,m20,m08=met(p1,gt),met(p20,gt),met(p08,gt);chg=p08&~p20;useful=int((chg & gt).sum());extra=int((chg&~gt).sum())
   frames.append(dict(row=r,i=i,err=float(m20['fn']),task=r['md']['task_id'],split=r['md']['split'],m1=m1,m20=m20,m08=m08,sw=int(chg.sum()),useful=useful,extra=extra,gt=gt,p1=p1,p20=p20,p08=p08))
 # 12 task cases: task-wise median V2@.20 FN. Then six rule-based distinct extras.
 chosen=[]
 for task in sorted(set(f['task'] for f in frames)):
  cand=[f for f in frames if f['task']==task];cand.sort(key=lambda x:(x['err'],x['row']['md']['demo'],int(x['row']['z']['state_indices'][x['i']])))
  chosen.append(cand[len(cand)//2])
 role_extra=[]
 for role in ('manipulated_objects','destination_objects'):
  vals=[]
  for r in rows:
   if role not in r['md']['role_keys']:continue
   ri=r['md']['role_keys'].index(role)
   for i in range(32):
    gt=r['z']['role_gt_occupancy'][i,ri].astype(np.float32)>0;pred=mask(r['z']['v2_probabilities'][i],.20); vals.append((int((gt&~pred).sum()),r,i,gt))
  vals.sort(key=lambda x:x[0],reverse=True)
  for n in range(2):
   _,r,i,_=next(v for v in vals if not any(c['row'] is v[1] and c['i']==v[2] for c in chosen+role_extra))
   role_extra.append(next(f for f in frames if f['row'] is r and f['i']==i))
 chosen+=role_extra
 # V1 better and high recall / low reuse cases on heldout frames.
 cands=[f for f in frames if f['split']=='held_out' and all(not(f['row'] is x['row'] and f['i']==x['i']) for x in chosen)]
 better=max(cands,key=lambda f:f['m1']['recall']-f['m20']['recall']); chosen.append(better)
 high=max((f for f in cands if f is not better),key=lambda f:f['m08']['recall']-f['m08']['reusable']/256);chosen.append(high)
 metric_rows=[]; manifest_rows=[]
 for n,f in enumerate(chosen,1):
  r=f['row'];ti=f['i'];state=int(r['z']['state_indices'][ti]);name=f"{n:02d}_{r['md']['task_id'].replace('/','_')}_{r['md']['demo']}_s{state:03d}.png"
  panel(r,ti,gallery/'gallery'/name)
  metric_rows.append(dict(example=n,split=f['split'],suite=r['task']['suite'],task=f['task'],demo=r['md']['demo'],frame_state=state,selection=('task_median_fn' if n<=12 else 'worst_role_fn' if n<=16 else 'v1_beats_v2' if n==17 else 'high_recall_low_reuse'),gt_fg_tokens=int(f['gt'].sum()),threshold_switch_tokens=f['sw'],switch_useful_corrections=f['useful'],switch_extra_background_protection=f['extra'],**{f'{k}_{x}':v for k in ('V1','V2@0.20','V2@0.08') for x,v in f[{'V1':'m1','V2@0.20':'m20','V2@0.08':'m08'}[k]].items()}))
  manifest_rows.append(dict(file=name,task=f['task'],split=f['split'],demo=r['md']['demo'],frame_state=state,cache_sha256=sha(r['cache']),hdf5_sha256=r['task']['hdf5_sha256'],selection=metric_rows[-1]['selection']))
 with (gallery/'example_metrics.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(metric_rows[0]));w.writeheader();w.writerows(metric_rows)
 (gallery/'gallery_manifest.json').write_text(json.dumps(dict(e2_manifest_sha256=sha(out/'libero_task_manifest.json'),role_audit_sha256=sha(out/'ground_truth_role_audit.json'),examples=manifest_rows),indent=2))
 # Three temporal strips, one held-out task per suite; R5 ages at consecutive samples.
 seqs=[]
 for suite in ('libero_spatial','libero_object','libero_goal'):
  seqs.append(next(r for r in rows if r['task']['suite']==suite and r['md']['split']=='held_out' and r['md']['demo']=='demo_0'))
 for q,r in enumerate(seqs,1):
  fig,axs=plt.subplots(4,8,figsize=(20,9),constrained_layout=True);fig.suptitle(f"Temporal ground-truth overlap proxy · {r['md']['task_id']} · {r['md']['demo']} (not KV correctness)")
  for age in range(1,5):
   for col in range(8):
    cur=age+col*3;src=cur-age;ax=axs[age-1,col];rgb=extract_rgb(r, int(r['z']['state_indices'][cur]));gtc=r['gt'][cur]>0;gts=r['gt'][src]>0
    p=mask(r['z']['v2_probabilities'][cur],.20); reused=~p; danger=reused & gtc; gain=gtc&~gts;lost=gts&~gtc
    # Dark=current GT safely classified BG; green protected GT; red reused GT; cyan gained; orange lost.
    view=np.zeros((16,16,4),np.float32);view[(~gtc&reused).reshape(16,16)]=(.08,.08,.08,.45);view[(p & gtc).reshape(16,16)]=(0,.8,.3,.8);view[danger.reshape(16,16)]=(1,0,0,.9);view[gain.reshape(16,16)]=(0.1,.7,1,.8);view[lost.reshape(16,16)]=(1,.6,0,.8)
    ax.imshow(rgb);ax.imshow(view,interpolation='nearest',extent=(0,rgb.shape[1],rgb.shape[0],0));ax.set_title(f"age {age} · state {int(r['z']['state_indices'][cur])}\nreuse∩GT {int(danger.sum())} / FG gain {int(gain.sum())}",fontsize=7);ax.axis('off')
  fig.savefig(gallery/'temporal'/f'{q}_{r["task"]["suite"]}.png',dpi=130);plt.close(fig)
 # Performance plots, all heldout frames.
 held=[f for f in frames if f['split']=='held_out']
 fig,ax=plt.subplots(figsize=(8,6))
 for key,field,color in [('V1','m1','#4878cf'),('V2@.20','m20','#e45756'),('V2@.08','m08','#54a24b')]:ax.scatter([f[field]['reusable']/256 for f in held],[f[field]['recall'] for f in held],s=12,alpha=.35,label=key,c=color)
 ax.set(xlabel='Candidate reuse fraction',ylabel='Binary token foreground recall',title='Held-out frame recall vs candidate reuse');ax.legend();fig.savefig(gallery/'plots/recall_vs_reuse.png',dpi=160);plt.close(fig)
 roles=('manipulated_objects','destination_objects','robot_arm','robot_gripper','task_relevant_fixtures','robot_mount');fig,ax=plt.subplots(figsize=(11,5));
 for key,idx,color in [('V1',0,'#4878cf'),('V2@0.20',1,'#e45756'),('V2@0.08',2,'#54a24b')]:
  vals=[]
  for role in roles:
   tp=fn=0
   for r in rows:
    if r['md']['split']!='held_out' or role not in r['md']['role_keys']:continue
    ri=r['md']['role_keys'].index(role)
    for i in range(32):
     g=r['z']['role_gt_occupancy'][i,ri].astype(np.float32);p=mask(r['z'][('v1_probabilities' if key=='V1' else 'v2_probabilities')][i],THRESHOLDS[key]);tp+=(p*g).sum();fn+=((~p)*g).sum()
   vals.append(tp/(tp+fn) if tp+fn else np.nan)
  ax.plot(range(len(roles)),vals,'o-',label=key,color=color)
 ax.set_xticks(range(len(roles)),roles,rotation=25,ha='right');ax.set_ylim(0,1);ax.set_ylabel('Occupancy-weighted recall');ax.legend();ax.set_title('Held-out per-role foreground recall');fig.tight_layout();fig.savefig(gallery/'plots/per_role_recall.png',dpi=160);plt.close(fig)
 # Required diagnostic plots over the full held-out frame set.
 fig,axs=plt.subplots(1,2,figsize=(12,4))
 fg=[];bg=[]
 for f in held:
  sc=np.max(f['row']['z']['v2_probabilities'][f['i']].astype(np.float32),axis=0);fg.extend(sc[f['gt']>0]);bg.extend(sc[f['gt']<=0])
 axs[0].hist(bg,bins=35,alpha=.65,density=True,label='GT background');axs[0].hist(fg,bins=35,alpha=.65,density=True,label='GT task foreground');axs[0].axvline(.08,color='green',ls='--');axs[0].axvline(.20,color='red',ls='--');axs[0].legend();axs[0].set_title('V2 max-prompt score distributions')
 for key,field,color in [('V1','m1','#4878cf'),('V2@.20','m20','#e45756'),('V2@.08','m08','#54a24b')]:axs[1].hist([f[field]['fn'] for f in held],bins=range(0,257,8),histtype='step',lw=2,label=key,color=color)
 axs[1].set_title('False-background tokens per held-out frame');axs[1].set_xlabel('FN tokens');axs[1].legend();fig.tight_layout();fig.savefig(gallery/'plots/score_and_fn_distributions.png',dpi=160);plt.close(fig)
 fig,axs=plt.subplots(1,2,figsize=(12,4))
 task_names=sorted(set(f['task'] for f in held));
 for key,field,color in [('V1','m1','#4878cf'),('V2@.20','m20','#e45756'),('V2@.08','m08','#54a24b')]:
  vals=[]
  for task in task_names:
   fs=[f for f in held if f['task']==task]; vals.append(float(np.mean([x[field]['recall'] for x in fs])))
  axs[0].plot(range(len(task_names)),vals,'o-',label=key,color=color)
 axs[0].set_xticks(range(len(task_names)),[x.split('/',1)[1][:20] for x in task_names],rotation=70,ha='right',fontsize=7);axs[0].set_ylim(0,1);axs[0].set_title('Held-out binary recall by task');axs[0].legend()
 useful=sum(f['useful'] for f in held);extra=sum(f['extra'] for f in held);axs[1].bar(['Useful corrections','Extra BG protected'],[useful,extra],color=['#00b95c','#ff9900']);axs[1].set_title('Threshold .20→.08 reclassification (held-out)');fig.tight_layout();fig.savefig(gallery/'plots/task_and_reclassification.png',dpi=160);plt.close(fig)
 # Role panels: binary ground-truth occupancy and whole-prompt-union predictions,
 # with roles lacking a dedicated prompt explicitly marked as unprompted.
 for n,f in enumerate(chosen[:6],1):
  r=f['row'];ti=f['i'];rgb=extract_rgb(r,int(r['z']['state_indices'][ti]));fig,axs=plt.subplots(6,3,figsize=(10,15),constrained_layout=True);fig.suptitle(f"Role-specific diagnostic (union prediction, not role classifier) · {f['task']} · {r['md']['demo']} state {int(r['z']['state_indices'][ti])}")
  for ri,role in enumerate(ROLES):
   gtrole=r['roles'][role][ti]>0;axs[ri,0].imshow(rgb);axs[ri,0].imshow(gtrole.reshape(16,16),cmap='Greens',alpha=.65,interpolation='nearest',extent=(0,rgb.shape[1],rgb.shape[0],0));axs[ri,0].set_title(role+' GT')
   for ci,(key,prob,threshold) in enumerate((('V1',r['z']['v1_probabilities'][ti],.55),('V2@.20',r['z']['v2_probabilities'][ti],.20)),1):
    pred=mask(prob,threshold);axs[ri,ci].imshow(rgb);axs[ri,ci].imshow(pred.reshape(16,16),cmap='Reds',alpha=.45,interpolation='nearest',extent=(0,rgb.shape[1],rgb.shape[0],0));axs[ri,ci].set_title(key+' union')
   prompt_roles=[x['role'] for x in r['md']['prompt_specs']]
   if role=='robot_mount' or role not in prompt_roles: axs[ri,2].text(.03,.03,'No dedicated prompt; union only',transform=axs[ri,2].transAxes,color='yellow',fontsize=8)
   for ax in axs[ri]:ax.axis('off')
  fig.savefig(gallery/'gallery'/f'role_{n:02d}.png',dpi=130);plt.close(fig)
 # report embeds all selected panels + existing overall results and limitations.
 report=['# Visual performance and reclassification gallery','', 'Generated solely from corrected E2 score/occupancy caches and the matching recorded HDF5 `agentview_rgb` frames. No inference, replay, training, or policy execution was performed. The HDF5 files were SHA-256 checked against the frozen manifest. Role audit revision: `region-aware-bddl-v2`.','', 'Binary token statistics classify a GT patch positive when its fractional occupancy is >0. TP/FP/FN/TN are counts over 256 cells. E2 aggregate recall in the report is occupancy-weighted, so these panel metrics are not numerically identical. Reclassification at .08 vs .20 labels switched foreground cells: green = relevant GT correction; orange = GT background newly protected; red = still-missed GT; dark = correctly reusable background.','', '## Dataset-level results','', '| Held-out method | Occupancy-weighted foreground recall | False-background | Mean candidate reusable tokens | Background coverage |','|---|---:|---:|---:|---:|']
 metrics=json.loads((out/'libero_ground_truth_metrics.json').read_text())['held_out_task_metrics']
 for key in ('V1_epoch50_seven_templates','V2_epoch40_COCO_prompt_0_20','V2_epoch40_dev_98pct_recall'):
  v=metrics[key];role=v['role_metrics']['task_relevant_foreground'];reuse=v['foreground']; report.append(f"| {key} | {role['recall']:.4f} | {role['false_background_rate']:.4f} | {reuse['mean_reusable_tokens']:.1f}/256 | {reuse['mean_candidate_background_coverage']:.3f} |")
 report += ['', '## Selected frame gallery','']
 for n,m in enumerate(metric_rows,1): report += [f"### {n}. {m['selection']} — {m['suite']} / {m['task'].split('/',1)[1]} / {m['demo']} / state {m['frame_state']}", '', f"![Comparison panel](gallery/{manifest_rows[n-1]['file']})", '']
 report += ['## Temporal strips','', 'Colors: red = current GT foreground that threshold .20 would reuse; green = current GT foreground predicted fresh; cyan = GT foreground gained since the R5 source; orange = source GT foreground lost since source. The displayed predicted reuse is .20 only; summary CSV contains the E2 .08 vs .20 age tables. These are overlap proxies, not KV correctness.', '']
 for q,r in enumerate(seqs,1):report += [f"![Temporal strip {q}](temporal/{q}_{r['task']['suite']}.png)",'']
 report += ['## Performance plots','','![Recall versus reuse](plots/recall_vs_reuse.png)','','![Per-role recall](plots/per_role_recall.png)','','## Interpretation','', 'The 0.20 cutoff misses substantial manipulated and destination foreground on held-out tasks. The development-calibrated 0.08 cutoff recovers much of the union foreground by making most patches fresh, sharply reducing reusable coverage. The robot mount has no dedicated prompt and is not validated as safe background. Small/thin objects remain coarse at 16×16; high occupancy recall must not be read as accurate boundaries. The held-out scores had already been inspected during the role-mapping audit, as disclosed in the E2 report; the 0.08 point was selected on development tasks only.','', 'See `../LIBERO_GROUND_TRUTH_REPORT.md` for the complete role tables, source provenance, and R5 age summary. Per-example binary metrics are in `example_metrics.csv`; selected-frame and source hashes are in `gallery_manifest.json`.']
 (gallery/'VISUAL_PERFORMANCE_REPORT.md').write_text('\n'.join(report)+'\n')
 (gallery/'VISUAL_FINDINGS.md').write_text('# Visual findings\n\n- V2@0.20 leaves many target-object cells eligible for reuse on held-out frames.\n- V2@0.08 reduces those false-background cells but also leaves only about 25 of 256 cells reusable on average.\n- Lowering the cutoff adds both useful foreground corrections and extra background protection; counts are per selected frame in `example_metrics.csv`.\n- Very small items and narrow gripper/fixture structures are poorly represented by a 16×16 grid.\n- Robot mount is unprompted and unvalidated; never treat its zero measured recall as safe background.\n- Temporal overlays show source/current mask overlap only, not actual cached K/V correctness or policy safety.\n')
 print('gallery complete',len(chosen),'panels',len(seqs),'temporal strips')

if __name__=='__main__':main()
