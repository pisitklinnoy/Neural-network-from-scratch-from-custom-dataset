"""Evaluate predicted lane masks against ground truth via pixel-wise IoU and metrics.

Metrics reported:
  - Detection Flag (Yes/No): IoU >= 0.6 is considered positive / detected.
  - Detection Rate: % of test set images achieving IoU >= 0.6.
  - Average IoU of Detected Images: Mean IoU over the detected positive subset.
  - Overall Mean IoU (mIoU): Mean IoU over all test images.
  - Mean Dice Coefficient / F1-Score.
  - Pixel-wise Precision, Recall, and Accuracy.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from dataset import list_images, load_yolo_seg_polygons, polygons_to_mask, label_path_for_image


def compute_binary_metrics(pred_mask, gt_mask, eps=1e-6):
    pred = (pred_mask > 0).astype(np.bool_)
    gt = (gt_mask > 0).astype(np.bool_)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()

    intersection = tp
    union = tp + fp + fn

    if union == 0:
        iou = 1.0 if intersection == 0 else 0.0
    else:
        iou = float(intersection) / float(union + eps)

    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate lane masks against ground truth.")
    parser.add_argument("--images-dir", default="data/images/val", help="Evaluation images directory.")
    parser.add_argument("--labels-dir", default="data/labels/val", help="Ground truth labels directory.")
    parser.add_argument("--pred-masks-dir", default="result", help="Predicted mask PNGs directory.")
    parser.add_argument("--iou-threshold", type=float, default=0.6, help="Detection threshold (default 0.6).")
    parser.add_argument("--output-json", default="evaluation_metrics.json", help="Path to save evaluation JSON.")
    args = parser.parse_args()

    images = list_images(args.images_dir)
    if not images:
        print(f"No images found under {args.images_dir}")
        return

    results = []
    print(f"Evaluating {len(images)} images against ground truth in '{args.labels_dir}'...")

    for image_path in images:
        pred_path = Path(args.pred_masks_dir) / f"{image_path.stem}.png"
        if not pred_path.exists():
            print(f"Warning: missing prediction for {image_path.name}, skipping.")
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        h, w = image.shape[:2]

        label_path = label_path_for_image(image_path, args.labels_dir)
        polygons = load_yolo_seg_polygons(label_path, w, h)
        gt_mask = polygons_to_mask(polygons, w, h)

        pred_mask = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
        if pred_mask is None:
            print(f"Warning: could not read prediction {pred_path}, skipping.")
            continue

        if pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        metrics = compute_binary_metrics(pred_mask, gt_mask)
        detected = metrics["iou"] >= args.iou_threshold

        results.append({
            "image": image_path.name,
            "detected": detected,
            **metrics,
        })

    n = len(results)
    if n == 0:
        print("No matching prediction files evaluated!")
        return

    detected_results = [r for r in results if r["detected"]]
    detection_rate = (len(detected_results) / n) * 100.0

    avg_iou_detected = (sum(r["iou"] for r in detected_results) / len(detected_results)
                        if detected_results else 0.0)
    avg_iou_all = sum(r["iou"] for r in results) / n
    avg_dice = sum(r["dice"] for r in results) / n
    avg_prec = sum(r["precision"] for r in results) / n
    avg_rec = sum(r["recall"] for r in results) / n
    avg_acc = sum(r["accuracy"] for r in results) / n

    summary = {
        "num_images": n,
        "iou_threshold": args.iou_threshold,
        "num_detected": len(detected_results),
        "detection_rate_pct": round(detection_rate, 2),
        "avg_iou_detected": round(avg_iou_detected, 4),
        "mean_iou_all": round(avg_iou_all, 4),
        "mean_dice_f1": round(avg_dice, 4),
        "mean_precision": round(avg_prec, 4),
        "mean_recall": round(avg_rec, 4),
        "pixel_accuracy": round(avg_acc, 4),
    }

    print("\n" + "=" * 55)
    print("        LANE SEGMENTATION EVALUATION REPORT")
    print("=" * 55)
    print(f" Total Images Evaluated     : {n}")
    print(f" Detection Threshold (IoU)  : >= {args.iou_threshold}")
    print(f" Detected Images (Positive) : {len(detected_results)} / {n}")
    print(f" Detection Rate             : {detection_rate:.2f}%")
    print(f" Average IoU (Detected Only): {avg_iou_detected:.4f}")
    print(f" Mean IoU (All Images)      : {avg_iou_all:.4f}")
    print(f" Mean Dice / F1-Score       : {avg_dice:.4f}")
    print(f" Mean Precision             : {avg_prec:.4f}")
    print(f" Mean Recall                : {avg_rec:.4f}")
    print(f" Mean Pixel Accuracy        : {avg_acc * 100:.2f}%")
    print("=" * 55 + "\n")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "per_image": results}, f, indent=2)
        print(f"Saved evaluation metrics to '{args.output_json}'")


if __name__ == "__main__":
    main()
