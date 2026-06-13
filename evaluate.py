"""
evaluate.py
-----------
Run after training to get detailed metrics on the test set:
    python evaluate.py

Outputs:
  - Per-class precision / recall / F1
  - Confusion matrix heatmap saved to outputs/
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import config
from dataset import build_datasets, build_loaders
from model import build_model, load_checkpoint


def evaluate():
    # 1. Reload test data
    train_ds, val_ds, test_ds, class_names, num_classes = build_datasets()
    _, _, test_loader = build_loaders(train_ds, val_ds, test_ds)

    # 2. Load best model
    model = build_model(num_classes, pretrained=False)
    checkpoint = load_checkpoint(model, config.CHECKPOINT_PATH)
    model.eval()

    all_preds, all_labels = [], []

    print("\nRunning evaluation on test set ...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(config.DEVICE)
            outputs = model(images)
            preds   = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 3. Classification report
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    # 4. Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig_size = max(12, num_classes // 2)      # scale with number of classes
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    sns.heatmap(
        cm, annot=(num_classes <= 30),        # hide numbers if too many classes
        fmt="d", cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True",      fontsize=12)
    ax.set_title("Confusion Matrix — Test Set", fontsize=14)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0,  fontsize=8)
    plt.tight_layout()
    save_path = f"{config.OUTPUT_DIR}/confusion_matrix.png"
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"Confusion matrix saved → {save_path}")

    # 5. Overall accuracy
    acc = (all_preds == all_labels).mean()
    print(f"\nTest Accuracy: {acc:.4f}  ({acc*100:.2f}%)")
    print(f"Best val acc (from checkpoint): {checkpoint.get('val_acc', 'N/A'):.4f}")


if __name__ == "__main__":
    evaluate()
