"""Run lane segmentation inference, measure memory footprint, and save results.

Features:
  - Generates full-resolution binary masks (1280x720, values 0 or 255).
  - Saves masks to disk (default: ./result or ./run/<run_name>/masks).
  - Measures inference memory footprint (VRAM, CPU RAM, model size, FLOPs/latency).
  - Generates snapshot comparisons (Before/After & Overlay) for assignment report.
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from dataset import list_images
from model import UNet


def load_model(checkpoint_path, device):
    model = UNet(in_channels=3, num_classes=1)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = state["model_state_dict"] if "model_state_dict" in state else state
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def preprocess(image_rgb, img_size):
    resized = cv2.resize(image_rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return tensor


@torch.no_grad()
def predict_mask(model, image_path, img_size, device, threshold=0.5):
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]

    tensor = preprocess(image_rgb, img_size).to(device)
    logits = model(tensor)
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

    mask_small = (prob > threshold).astype(np.uint8) * 255
    mask_full = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask_full, image_bgr


def measure_memory_footprint(model, device, img_size=48):
    """Measures model parameters, weights memory, and peak inference VRAM."""
    n_params = sum(p.numel() for p in model.parameters())
    weights_mem_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)

    dummy_input = torch.randn(1, 3, img_size, img_size, device=device)

    # Warmup
    for _ in range(5):
        _ = model(dummy_input)

    peak_vram_mb = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
        with torch.no_grad():
            _ = model(dummy_input)
        torch.cuda.synchronize()
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

    # Latency benchmark
    timings = []
    with torch.no_grad():
        for _ in range(30):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0)

    avg_latency_ms = float(np.mean(timings))
    fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0

    return {
        "parameters": n_params,
        "weights_memory_mb": weights_mem_mb,
        "peak_vram_mb": peak_vram_mb,
        "avg_latency_ms": avg_latency_ms,
        "fps": fps,
    }


def create_snapshot_comparison(image_bgr, mask_full, out_path):
    """Creates a side-by-side snapshot: Original | Predicted Mask | Overlay."""
    h, w = image_bgr.shape[:2]

    # Convert grayscale mask to 3-channel
    mask_vis = cv2.cvtColor(mask_full, cv2.COLOR_GRAY2BGR)

    # Overlay
    overlay = image_bgr.copy()
    lane_idx = mask_full > 0
    green = np.zeros_like(image_bgr)
    green[:, :] = (0, 220, 0)
    overlay[lane_idx] = cv2.addWeighted(image_bgr[lane_idx], 0.45, green[lane_idx], 0.55, 0)

    # Put labels on top
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.9
    thick = 2
    color = (255, 255, 255)

    img_label = image_bgr.copy()
    cv2.putText(img_label, "1. Input Camera Frame", (30, 45), font, scale, color, thick, cv2.LINE_AA)
    cv2.putText(mask_vis, "2. Binary Mask (1280x720)", (30, 45), font, scale, color, thick, cv2.LINE_AA)
    cv2.putText(overlay, "3. Lane Segmentation Overlay", (30, 45), font, scale, color, thick, cv2.LINE_AA)

    # Resize for nice horizontal display
    disp_w, disp_h = 640, 360
    c1 = cv2.resize(img_label, (disp_w, disp_h))
    c2 = cv2.resize(mask_vis, (disp_w, disp_h))
    c3 = cv2.resize(overlay, (disp_w, disp_h))

    composite = np.hstack([c1, c2, c3])
    cv2.imwrite(str(out_path), composite)


def main():
    parser = argparse.ArgumentParser(description="Lane segmentation inference and memory profiling.")
    parser.add_argument("--images-dir", default="data/images/val", help="Path to input images directory.")
    parser.add_argument("--checkpoint", default="checkpoints/unet_lane/best.pt", help="Path to model checkpoint.")
    parser.add_argument("--img-size", type=int, default=48)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", default="result", help="Destination directory for binary masks.")
    parser.add_argument("--snapshots-dir", default="snapshots", help="Destination directory for comparison snapshots.")
    parser.add_argument("--num-snapshots", type=int, default=5, help="Number of snapshot comparisons to save.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==================================================")
    print(f" LANE SEGMENTATION INFERENCE & PROFILING")
    print(f" Device: {device}")
    if device.type == "cuda":
        print(f" GPU: {torch.cuda.get_device_name(0)}")
    print(f"==================================================")

    model = load_model(args.checkpoint, device)

    # Measure memory footprint
    mem_stats = measure_memory_footprint(model, device, args.img_size)
    print("\n--- Memory Footprint & Efficiency Report ---")
    print(f"  Model Parameters      : {mem_stats['parameters']:,}")
    print(f"  Model Weights Memory  : {mem_stats['weights_memory_mb']:.2f} MB")
    if device.type == "cuda":
        print(f"  Peak GPU VRAM (infer) : {mem_stats['peak_vram_mb']:.2f} MB")
    print(f"  Inference Latency     : {mem_stats['avg_latency_ms']:.2f} ms/frame")
    print(f"  Inference Speed       : {mem_stats['fps']:.1f} FPS")
    print("--------------------------------------------\n")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    snap_dir = Path(args.snapshots_dir)
    snap_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(args.images_dir)
    if not images:
        print(f"No images found under {args.images_dir}")
        return

    print(f"Running inference on {len(images)} images from {args.images_dir}...")
    for idx, image_path in enumerate(images):
        mask, image_bgr = predict_mask(model, image_path, args.img_size, device, args.threshold)
        out_path = out_dir / f"{image_path.stem}.png"
        cv2.imwrite(str(out_path), mask)

        if idx < args.num_snapshots:
            snap_path = snap_dir / f"snapshot_{idx+1}_{image_path.stem}.jpg"
            create_snapshot_comparison(image_bgr, mask, snap_path)

    print(f"\n[DONE] Saved {len(images)} binary masks to '{out_dir}'")
    print(f"[DONE] Saved {min(len(images), args.num_snapshots)} comparison snapshots to '{snap_dir}'")


if __name__ == "__main__":
    main()
