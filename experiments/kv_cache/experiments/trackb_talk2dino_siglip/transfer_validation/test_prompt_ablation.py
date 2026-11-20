import torch
import torch.nn as nn
from .prompt_ablation import METHODS, choose_thresholds, make_probabilities

class Projection(nn.Module):
    def __init__(self):
        super().__init__(); self.text_projection=nn.Linear(5,4,bias=False)

def test_prompt_conditions_shape_and_ensembles():
    torch.manual_seed(21)
    v1,v2=Projection(),Projection()
    p1=torch.randn(256,4);p2=torch.randn(256,4)
    four=torch.randn(4,3,5);tv1=torch.randn(3,5)
    results,separate=make_probabilities(v1,v2,p1,p2,four,tv1)
    assert set(results)==set(METHODS)
    assert all(x.shape==(3,256) and torch.isfinite(x).all() for x in results.values())
    assert torch.allclose(results["F_score_ensemble"],separate.mean(0))
    assert separate.shape==(4,3,256)

def test_dev_threshold_selection_deterministic_including_v1_calibration():
    rec={"categories":{"1":{"methods":{}}}}
    for method in METHODS:
        values={f"{x:.2f}":[8,2,2] if x==.45 else [3,7,7] for x in [i/100 for i in range(5,96,5)]}
        rec["categories"]["1"]["methods"][method]={"threshold_sweep_tp_fp_fn":values}
    thresholds,details=choose_thresholds([rec])
    assert thresholds["A_training_prompt"]==.45
    assert thresholds["V1_development_recalibrated"]==.45
    assert details["V1_development_recalibrated"]["derived_from"]=="V1 seven-template development sweep"
