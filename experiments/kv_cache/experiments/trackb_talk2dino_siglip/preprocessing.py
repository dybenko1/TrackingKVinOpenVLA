"""Talk2DINO's independent DINO image transform; OpenVLA preprocessing is untouched."""
from __future__ import annotations

from typing import Any


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TALK2DINO_IMAGE_SIZE = 224


def build_talk2dino_transform(image_size: int = TALK2DINO_IMAGE_SIZE) -> Any:
    """Resize shorter side bicubically, center-crop, tensorize, ImageNet-normalize."""
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    if image_size <= 0:
        raise ValueError("image_size must be positive")
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def preprocess_talk2dino_image(image: Any, transform: Any | None = None):
    """Preprocess a PIL RGB image on the sidecar path only."""
    import torch

    if transform is None:
        transform = build_talk2dino_transform()
    tensor = transform(image.convert("RGB"))
    if tensor.ndim != 3 or tuple(tensor.shape[-2:]) != (TALK2DINO_IMAGE_SIZE, TALK2DINO_IMAGE_SIZE):
        raise AssertionError(f"Talk2DINO transform must yield [3,224,224], got {tuple(tensor.shape)}")
    if not torch.isfinite(tensor).all():
        raise FloatingPointError("Talk2DINO transform produced non-finite pixels")
    return tensor
