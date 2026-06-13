"""
predict.py
----------
Use the trained model to predict disease on a new leaf image.

Usage:
    python predict.py --image path/to/leaf.jpg
    python predict.py --folder path/to/test_images/
"""

import argparse
import torch
from PIL import Image
from pathlib import Path

import config
from dataset import get_val_transforms
from model import build_model, load_checkpoint


def predict_single(model, image_path, class_names):
    """Returns (predicted_class, confidence, top5_list)."""
    transform = get_val_transforms()
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    top5_probs, top5_idxs = probs.topk(min(5, len(class_names)))
    top5 = [(class_names[i], p.item()) for i, p in zip(top5_idxs, top5_probs)]

    best_class = top5[0][0]
    confidence = top5[0][1]
    return best_class, confidence, top5


def main():
    parser = argparse.ArgumentParser(description="Plant disease predictor")
    parser.add_argument("--image",  type=str, default=None, help="Path to a single image")
    parser.add_argument("--folder", type=str, default=None, help="Path to a folder of images")
    args = parser.parse_args()

    # Load model
    checkpoint  = torch.load(config.CHECKPOINT_PATH, map_location=config.DEVICE)
    class_names = checkpoint["class_names"]
    num_classes = checkpoint["num_classes"]

    model = build_model(num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Model loaded — {num_classes} classes")

    if args.image:
        cls, conf, top5 = predict_single(model, args.image, class_names)
        print(f"\nImage : {args.image}")
        print(f"Prediction : {cls}  ({conf*100:.1f}%)")
        print("Top-5:")
        for rank, (name, prob) in enumerate(top5, 1):
            print(f"  {rank}. {name:<50s}  {prob*100:.2f}%")

    elif args.folder:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        images = [p for p in Path(args.folder).rglob("*") if p.suffix.lower() in exts]
        print(f"\nFound {len(images)} images in '{args.folder}'\n")
        for img_path in images:
            cls, conf, _ = predict_single(model, img_path, class_names)
            print(f"{img_path.name:<40s} → {cls:<50s} ({conf*100:.1f}%)")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
