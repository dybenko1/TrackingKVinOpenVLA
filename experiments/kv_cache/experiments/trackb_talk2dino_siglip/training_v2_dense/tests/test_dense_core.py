import numpy as np
import torch
from pathlib import Path
from experiments.kv_cache.experiments.trackb_talk2dino_siglip.training_v2_dense.losses import dense_pair_losses, patch_logmeanexp_scores
from experiments.kv_cache.experiments.trackb_talk2dino_siglip.training_v2_dense.sampler import CaptionRoundSampler, choose_categories
from experiments.kv_cache.experiments.trackb_talk2dino_siglip.training_v2_dense.masks import fractional_grid
from experiments.kv_cache.experiments.trackb_talk2dino_siglip.training_v2_dense.evaluate import compute_maps
from experiments.kv_cache.experiments.trackb_talk2dino_siglip.projection import RegionTextProjection

def test_caption_round_sampler_full_coverage_and_uniqueness():
    ids=[1,1,1,2,2,3,4,4,5]
    sampler=CaptionRoundSampler(ids,3,123); sampler.set_epoch(0)
    batches=list(sampler); flat=[r for b in batches for r in b]
    assert sorted(flat)==list(range(len(ids)))
    assert all(len({ids[r] for r in b})==len(b) for b in batches)
    sampler.set_epoch(0); assert batches==list(sampler)
    sampler.set_epoch(1); assert batches!=list(sampler)

def test_fractional_grid_preserves_area_and_alignment():
    m=np.zeros((224,224),np.uint8);m[0:14,0:14]=1;m[14:28,14:28]=1
    g=fractional_grid(m,16).reshape(16,16)
    assert g[0,0]==1 and g[1,1]==1 and np.isclose(g.sum(),2)

def test_dense_losses_finite_and_gradient():
    z=torch.randn(5,1,256,requires_grad=True)
    y=torch.zeros_like(z);y[:,:,3:5]=torch.tensor([[[.2,.6]]])
    valid=torch.ones_like(z);valid[:,:,20]=0
    d=dense_pair_losses(z,y,valid,y.sum(-1)[:,0])
    loss=d["bce"]+d["dice"]+.1*d["hard_negative"]
    assert torch.isfinite(loss);loss.backward();assert torch.isfinite(z.grad).all()

def test_patch_pooling_dimensions_and_finite():
    x=torch.randn(4,256,1024);t=torch.randn(4,1024)
    s=patch_logmeanexp_scores(x,t)
    assert s.shape==(4,4) and torch.isfinite(s).all()

def test_category_selection_deterministic():
    areas=np.array([.2,8.,22.,1.]);freq=np.array([10,4,100,8])
    a=choose_categories(8,[0,1,2,3],areas,freq,2,0,4,123)
    b=choose_categories(8,[0,1,2,3],areas,freq,2,0,4,123)
    assert a==b and len(a)==4 and a[0]==0

def test_eval_output_mapping_all_256_and_three_conditions():
    torch.manual_seed(9)
    a=RegionTextProjection();b=RegionTextProjection()
    patch1=torch.randn(256,1024);patch2=torch.randn(256,1024);text=torch.randn(3,1152);rgb=torch.rand(3,224,224)
    maps224,maps16=compute_maps(a,b,patch1,patch2,text,rgb,lambda image,mask:mask)
    assert set(maps16)=={"v1_raw_pamr","v2_raw","v2_pamr"}
    assert all(x.shape==(3,16,16) for x in maps16.values())
    assert all(x.shape==(3,224,224) for x in maps224.values())
    assert all(torch.isfinite(x).all() for x in list(maps16.values())+list(maps224.values()))

def test_atomic_checkpoint_resume_preserves_optimizer_and_next_update(tmp_path):
    from experiments.kv_cache.experiments.trackb_talk2dino_siglip.training_v2_dense.train import atomic_torch
    torch.manual_seed(3); model=RegionTextProjection();opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
    x=torch.randn(4,1152);y=torch.randn(4,1024)
    loss=(model.text_projection(x)-y).square().mean();loss.backward();opt.step()
    state={"model":model.state_dict(),"optimizer":opt.state_dict(),"torch_rng":torch.get_rng_state()}
    path=tmp_path/"latest.pt";atomic_torch(path,state)
    uninterrupted=RegionTextProjection();uninterrupted.load_state_dict(model.state_dict());opt_a=torch.optim.AdamW(uninterrupted.parameters(),lr=1e-4);opt_a.load_state_dict(opt.state_dict())
    torch.set_rng_state(state["torch_rng"]);expected=torch.randn(4,1152); target=torch.randn(4,1024)
    la=(uninterrupted.text_projection(expected)-target).square().mean();opt_a.zero_grad();la.backward();opt_a.step()
    resumed=RegionTextProjection();opt_b=torch.optim.AdamW(resumed.parameters(),lr=1e-4)
    saved=torch.load(path,map_location="cpu",weights_only=False);resumed.load_state_dict(saved["model"]);opt_b.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["torch_rng"]);expected_b=torch.randn(4,1152);target_b=torch.randn(4,1024)
    lb=(resumed.text_projection(expected_b)-target_b).square().mean();opt_b.zero_grad();lb.backward();opt_b.step()
    assert torch.equal(expected,expected_b) and torch.equal(target,target_b)
    assert all(torch.equal(x,y) for x,y in zip(uninterrupted.parameters(),resumed.parameters()))
