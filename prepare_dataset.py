"""Prepare and split the 1,000-frame dataset_seg for Lane Segmentation.

Dataset: c:\\Assignment\\Lab_R\\Last_assignment\\dataset_seg
Split: 65% Train (650 frames) / 35% Val (350 frames)
Target Class in dataset_seg: Class 3 ('lane' road surface polygon)
Remaps target class to class 0 for clean single-class YOLO-seg format.
"""
import argparse
import os
import shutil
import random
from pathlib import Path


def prepare_dataset(
    images_dir=r"C:\Assignment\Lab_R\Last_assignment\dataset_seg\images\all_images",
    labels_dir=r"C:\Assignment\Lab_R\Last_assignment\dataset_seg\labels\all_images",
    output_dir=r"C:\Assignment\Lab_R\Last_assignment\data",
    train_ratio=0.65,
    seed=42,
    target_class=3,
):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)

    train_img_dir = output_dir / "images" / "train"
    val_img_dir = output_dir / "images" / "val"
    train_lbl_dir = output_dir / "labels" / "train"
    val_lbl_dir = output_dir / "labels" / "val"

    # Reset output directory cleanly
    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    img_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    valid_pairs = []

    for img_p in img_files:
        lbl_p = labels_dir / f"{img_p.stem}.txt"
        if lbl_p.exists():
            valid_pairs.append((img_p, lbl_p))

    print(f"Found {len(valid_pairs)} valid (image, label) pairs in dataset_seg.")
    if not valid_pairs:
        raise RuntimeError(f"No matching pairs found in {images_dir} and {labels_dir}")

    random.seed(seed)
    random.shuffle(valid_pairs)

    n_total = len(valid_pairs)
    n_train = int(n_total * train_ratio)
    train_pairs = valid_pairs[:n_train]
    val_pairs = valid_pairs[n_train:]

    print(f"Dataset split (ratio {train_ratio:.0%} / {1-train_ratio:.0%}):")
    print(f"  - Train: {len(train_pairs)} samples")
    print(f"  - Val / Test: {len(val_pairs)} samples")

    def process_and_copy(pairs, dest_img_dir, dest_lbl_dir):
        for img_p, lbl_p in pairs:
            shutil.copy2(img_p, dest_img_dir / img_p.name)

            filtered_lines = []
            with open(lbl_p, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 7:
                        cls_id = int(parts[0])
                        if cls_id == target_class:
                            new_line = "0 " + " ".join(parts[1:]) + "\n"
                            filtered_lines.append(new_line)

            out_lbl = dest_lbl_dir / lbl_p.name
            with open(out_lbl, "w", encoding="utf-8") as f:
                f.writelines(filtered_lines)

    print("Formatting and writing train split...")
    process_and_copy(train_pairs, train_img_dir, train_lbl_dir)

    print("Formatting and writing validation/test split...")
    process_and_copy(val_pairs, val_img_dir, val_lbl_dir)

    # Write data.yaml
    yaml_content = f"""path: {output_dir.as_posix()}
train: images/train
val: images/val
test: images/val

names:
  0: lane
"""
    with open(output_dir / "data.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"Done! Dataset ready in '{output_dir}'.")
    print(f"  Train images: {len(train_pairs)}")
    print(f"  Val images:   {len(val_pairs)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", default=r"C:\Assignment\Lab_R\Last_assignment\dataset_seg\images\all_images")
    parser.add_argument("--labels-dir", default=r"C:\Assignment\Lab_R\Last_assignment\dataset_seg\labels\all_images")
    parser.add_argument("--output-dir", default=r"C:\Assignment\Lab_R\Last_assignment\data")
    parser.add_argument("--train-ratio", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-class", type=int, default=3)
    args = parser.parse_args()

    prepare_dataset(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        seed=args.seed,
        target_class=args.target_class,
    )
