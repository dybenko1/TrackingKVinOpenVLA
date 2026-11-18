from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DenseV2Config:
    seed: int = 123
    image_size: int = 224
    grid_size: int = 16
    patch_dim: int = 1024
    text_dim: int = 1152
    hidden_dim: int = 1024
    max_epochs: int = 40
    min_epochs: int = 12
    evaluate_every: int = 2
    patience_events: int = 5
    improvement_tolerance: float = 0.001
    batch_size: int = 128
    lr: float = 1e-4
    min_lr: float = 1e-6
    weight_decay: float = 0.01
    warmup_fraction: float = 0.02
    grad_clip: float = 1.0
    temperature: float = 0.07
    pool_temperature: float = 0.10
    dense_logit_scale: float = 10.0
    dense_logit_bias: float = 0.20
    positive_weight: float = 4.0
    hard_negative_weight: float = 0.10
    bce_weight: float = 0.50
    dice_weight: float = 0.50
    infonce_weight: float = 0.20
    masks_per_image_visit: int = 4
    hard_negatives: int = 16
    occupancy_negative_cutoff: float = 0.02

    def to_dict(self):
        return asdict(self)
