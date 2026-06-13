import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

import config
from dataset import build_datasets, build_loaders
from model import build_model, freeze_backbone, unfreeze_last_blocks, save_checkpoint


def run_epoch(model, loader, criterion, optimizer, scaler, scheduler, phase="train"):
    is_train = (phase == "train")
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(is_train):
        for images, labels in tqdm(loader, desc=f"  {phase}", leave=False):
            images = images.to(config.DEVICE, non_blocking=True)
            labels = labels.to(config.DEVICE, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=config.MIXED_PRECISION):
                outputs = model(images)
                loss    = criterion(outputs, labels)

            if is_train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            total_loss += loss.item() * images.size(0)
            preds       = outputs.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += images.size(0)

    return total_loss / total, correct / total


def build_optimizer_and_scheduler(model, num_steps_per_epoch):
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE * 0.1,
        weight_decay=config.WEIGHT_DECAY,
    )

    total_steps = config.NUM_EPOCHS * num_steps_per_epoch

    def lr_lambda(step):
        progress = step / max(total_steps, 1)
        return max(0.0, 0.5 * (1 + torch.cos(torch.tensor(3.14159 * progress)).item()))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def plot_curves(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"],   label="Val")
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend()

    ax2.plot(epochs, history["train_acc"], label="Train")
    ax2.plot(epochs, history["val_acc"],   label="Val")
    ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"[Plot]   Curves saved -> {save_path}")


def train():
    print(f"\n{'='*55}")
    print(f"  Plant Disease Classifier - {config.MODEL_NAME}")
    print(f"  Device : {config.DEVICE}")
    print(f"  Epochs : {config.NUM_EPOCHS} | Batch : {config.BATCH_SIZE}")
    print(f"{'='*55}\n")

    # 1. Data
    train_ds, val_ds, test_ds, class_names, num_classes = build_datasets()
    train_loader, val_loader, test_loader = build_loaders(train_ds, val_ds, test_ds)

    # 2. Model
    model = build_model(num_classes)

    # 3. Resume from checkpoint
    start_epoch  = 6
    best_val_acc = 0.8317

    if os.path.exists(config.LAST_CHECKPOINT):
        print(f"[Resume] Found checkpoint -> '{config.LAST_CHECKPOINT}'")
        checkpoint   = torch.load(config.LAST_CHECKPOINT, map_location=config.DEVICE)
        model.load_state_dict(checkpoint["model_state"])
        start_epoch  = checkpoint.get("epoch", 5) + 1
        best_val_acc = checkpoint.get("val_acc", 0.8317)
        print(f"[Resume] Resuming from epoch {start_epoch} | Best acc: {best_val_acc:.4f}")
    elif os.path.exists(config.CHECKPOINT_PATH):
        print(f"[Resume] Loading best model weights -> '{config.CHECKPOINT_PATH}'")
        checkpoint   = torch.load(config.CHECKPOINT_PATH, map_location=config.DEVICE)
        model.load_state_dict(checkpoint["model_state"])
        print(f"[Resume] Loaded. Starting from epoch {start_epoch}")
    else:
        print(f"[Warning] No checkpoint found. Starting fresh from epoch {start_epoch}")

    # 4. Unfreeze only last 3 blocks + classifier head
    #    Saves VRAM — allows batch size 24 safely on RTX 3050
    unfreeze_last_blocks(model, num_blocks=3)

    # 5. Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)

    # 6. Optimizer + Scheduler + Scaler
    optimizer, scheduler = build_optimizer_and_scheduler(model, len(train_loader))
    scaler = torch.amp.GradScaler('cuda', enabled=config.MIXED_PRECISION)

    # 7. Training loop
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    patience_count = 0

    print(f"\n[Training] Epoch {start_epoch} to {config.NUM_EPOCHS}")
    print(f"[Training] Best val accuracy to beat: {best_val_acc:.4f}")
    print(f"[Training] Batch size: {config.BATCH_SIZE} | Mixed precision: {config.MIXED_PRECISION}\n")

    for epoch in range(start_epoch, config.NUM_EPOCHS + 1):
        epoch_start = time.time()

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, scaler, scheduler, "train")
        val_loss,   val_acc   = run_epoch(model, val_loader,   criterion, optimizer, scaler, scheduler, "val")

        elapsed    = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch:>3}/{config.NUM_EPOCHS}] "
            f"Train loss: {train_loss:.4f}  acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f}  acc: {val_acc:.4f} | "
            f"LR: {current_lr:.2e}  [{elapsed:.1f}s]"
        )

        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            patience_count = 0
            save_checkpoint(model, epoch, val_acc, config.CHECKPOINT_PATH, {
                "class_names": class_names,
                "num_classes": num_classes,
            })
            print(f"  New best val accuracy: {val_acc:.4f} - checkpoint saved")
        else:
            patience_count += 1
            print(f"  No improvement. Patience: {patience_count}/{config.PATIENCE}")
            if patience_count >= config.PATIENCE:
                print(f"\n[Early stop] No improvement for {config.PATIENCE} epochs. Stopping.")
                break

        save_checkpoint(model, epoch, val_acc, config.LAST_CHECKPOINT)

    print(f"\n{'='*55}")
    print(f"  Training complete.  Best val accuracy: {best_val_acc:.4f}")
    print(f"  Best model saved -> {config.CHECKPOINT_PATH}")
    print(f"{'='*55}\n")

    plot_curves(history, f"{config.OUTPUT_DIR}/training_curves.png")


if __name__ == "__main__":
    train()