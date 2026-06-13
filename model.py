"""
model.py
--------
Builds an EfficientNet-B3 (or any timm model) with a custom classification head.
Strategy:
  Phase 1 — freeze backbone, train only the head for ~5 epochs
  Phase 2 — unfreeze all layers and fine-tune at a lower LR
"""

import timm
import torch
import torch.nn as nn
import config


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Creates a pretrained EfficientNet-B3 with its classifier replaced
    to predict `num_classes` plant disease categories.
    """
    model = timm.create_model(
        config.MODEL_NAME,
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model.to(config.DEVICE)


def freeze_backbone(model: nn.Module):
    """
    Freeze all layers except the final classifier head.
    Call this at the start of training.
    """
    for name, param in model.named_parameters():
        if "classifier" not in name:       # timm uses 'classifier' for the head
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[Model]  Backbone FROZEN — trainable params: {trainable:,} / {total:,}")


def unfreeze_all(model: nn.Module):
    """
    Unfreeze every parameter for full fine-tuning.
    Call this after the warm-up phase.
    """
    for param in model.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model]  All layers UNFROZEN — trainable params: {trainable:,}")

def unfreeze_last_blocks(model: nn.Module, num_blocks: int = 3):
    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True

    for name, param in model.named_parameters():
        for i in range(7 - num_blocks, 7):
            if f"blocks.{i}" in name:
                param.requires_grad = True

    for name, param in model.named_parameters():
        if "conv_head" in name or "bn2" in name:
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[Model]  Last {num_blocks} blocks UNFROZEN — trainable: {trainable:,} / {total:,}")


def load_checkpoint(model: nn.Module, path: str):
    """Loads a saved checkpoint into the model (weights only)."""
    checkpoint = torch.load(path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    print(f"[Model]  Loaded checkpoint from '{path}'  (epoch {checkpoint.get('epoch', '?')})")
    return checkpoint


def save_checkpoint(model: nn.Module, epoch: int, val_acc: float, path: str, extra: dict = None):
    """Saves model weights + metadata to disk."""
    payload = {
        "epoch":       epoch,
        "val_acc":     val_acc,
        "model_state": model.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
