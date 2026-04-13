"""
Baseball Detection - YOLOv8 Training Pipeline
Econ 8310 - Semester Project
Author: Steph Simmons

This script:
1. Parses your CVAT XML annotations
2. Converts them to YOLO format
3. Splits data into train/val sets
4. Fine-tunes a pre-trained YOLOv8 model
5. Evaluates and saves results

Requirements (run setup.py first to install):
    pip install ultralytics opencv-python matplotlib

Usage:
    python train.py
"""

import os
import xml.etree.ElementTree as ET
import shutil
import random
import yaml
from pathlib import Path

# ─────────────────────────────────────────
# CONFIGURATION — update these paths
# ─────────────────────────────────────────
CVAT_XML     = r"C:\Users\req88455\Downloads\simmonscvat\simmonscvat.xml"
FRAMES_DIR   = r"C:\Users\req88455\Downloads\Videos"   # folder containing your 7 video frame subfolders
OUTPUT_DIR   = r"C:\Users\req88455\Downloads\baseball_yolo"  # where to save dataset + results
MODEL_SIZE   = "yolov8n.pt"   # nano = fastest on CPU; options: yolov8n/s/m/l/x
EPOCHS       = 50
IMG_SIZE     = 640
VAL_SPLIT    = 0.2            # 20% of frames used for validation
RANDOM_SEED  = 42
# ─────────────────────────────────────────


def parse_cvat_xml(xml_path):
    """
    Parse CVAT 1.1 XML and return list of:
        (image_name, full_image_path, [(x1,y1,x2,y2), ...])
    Only includes images that have at least one bounding box.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    annotations = []
    missing = 0

    for image_elem in root.findall("image"):
        img_name = image_elem.get("name")          # e.g. IMG_8226_jared/frame_0001.jpg
        img_w    = int(image_elem.get("width"))
        img_h    = int(image_elem.get("height"))

        boxes = []
        for box in image_elem.findall("box"):
            xtl = float(box.get("xtl"))
            ytl = float(box.get("ytl"))
            xbr = float(box.get("xbr"))
            ybr = float(box.get("ybr"))
            boxes.append((xtl, ytl, xbr, ybr, img_w, img_h))

        if not boxes:
            continue  # skip unannotated frames

        # Build full path to the image file
        # The XML stores paths like "IMG_8226_jared/frame_0001.jpg"
        img_subpath = img_name.replace("/", os.sep)
        img_full    = os.path.join(FRAMES_DIR, img_subpath)

        if not os.path.exists(img_full):
            missing += 1
            continue

        annotations.append((img_name, img_full, boxes))

    print(f"Parsed {len(annotations)} annotated frames ({missing} image files not found)")
    return annotations


def cvat_box_to_yolo(xtl, ytl, xbr, ybr, img_w, img_h):
    """
    Convert CVAT absolute pixel coords to YOLO normalized format:
        class_id  cx  cy  w  h   (all normalized 0-1)
    Class 0 = baseball
    """
    cx = ((xtl + xbr) / 2) / img_w
    cy = ((ytl + ybr) / 2) / img_h
    w  = (xbr - xtl) / img_w
    h  = (ybr - ytl) / img_h
    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def build_yolo_dataset(annotations):
    """
    Copy images and write YOLO label .txt files into:
        OUTPUT_DIR/
            images/train/   images/val/
            labels/train/   labels/val/
    """
    random.seed(RANDOM_SEED)
    random.shuffle(annotations)

    n_val   = int(len(annotations) * VAL_SPLIT)
    val_set = annotations[:n_val]
    trn_set = annotations[n_val:]

    for split, data in [("train", trn_set), ("val", val_set)]:
        img_dir = Path(OUTPUT_DIR) / "images" / split
        lbl_dir = Path(OUTPUT_DIR) / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_name, img_full, boxes in data:
            # Unique filename: replace path separators with underscore
            safe_name = img_name.replace("/", "_").replace("\\", "_")
            stem      = Path(safe_name).stem   # no extension

            # Copy image
            dst_img = img_dir / safe_name
            shutil.copy2(img_full, dst_img)

            # Write YOLO label file
            dst_lbl = lbl_dir / f"{stem}.txt"
            with open(dst_lbl, "w") as f:
                for (xtl, ytl, xbr, ybr, iw, ih) in boxes:
                    f.write(cvat_box_to_yolo(xtl, ytl, xbr, ybr, iw, ih) + "\n")

    print(f"Dataset built: {len(trn_set)} train, {len(val_set)} val frames")
    return len(trn_set), len(val_set)


def write_yaml():
    """Write the dataset YAML config that YOLOv8 needs."""
    yaml_path = Path(OUTPUT_DIR) / "dataset.yaml"
    config = {
        "path"  : str(OUTPUT_DIR),
        "train" : "images/train",
        "val"   : "images/val",
        "nc"    : 1,
        "names" : ["baseball"]
    }
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Dataset YAML written: {yaml_path}")
    return str(yaml_path)


def train_model(yaml_path):
    """Fine-tune YOLOv8 on our baseball dataset."""
    from ultralytics import YOLO

    print(f"\nLoading pre-trained model: {MODEL_SIZE}")
    model = YOLO(MODEL_SIZE)

    print(f"Starting training: {EPOCHS} epochs, image size {IMG_SIZE}")
    print("This will take a while on CPU — go get a coffee!\n")

    results = model.train(
        data      = yaml_path,
        epochs    = EPOCHS,
        imgsz     = IMG_SIZE,
        batch     = 4,          # small batch for CPU
        device    = "cpu",
        project   = str(Path(OUTPUT_DIR) / "runs"),
        name      = "baseball_detect",
        patience  = 15,         # early stopping if no improvement
        save      = True,
        plots     = True,
        verbose   = True,
    )
    return results


def evaluate_model():
    """Run validation on the best saved model and print metrics."""
    from ultralytics import YOLO

    best_model_path = Path(OUTPUT_DIR) / "runs" / "baseball_detect" / "weights" / "best.pt"
    if not best_model_path.exists():
        print("No trained model found — run training first.")
        return

    print(f"\nEvaluating best model: {best_model_path}")
    model   = YOLO(str(best_model_path))
    metrics = model.val(
        data   = str(Path(OUTPUT_DIR) / "dataset.yaml"),
        imgsz  = IMG_SIZE,
        device = "cpu",
    )

    print("\n========== RESULTS ==========")
    print(f"mAP@0.5:       {metrics.box.map50:.4f}")
    print(f"mAP@0.5-0.95:  {metrics.box.map:.4f}")
    print(f"Precision:     {metrics.box.mp:.4f}")
    print(f"Recall:        {metrics.box.mr:.4f}")
    print("=============================")
    print("\nThese metrics tell you how well the model detects baseballs:")
    print("  mAP@0.5 = detection accuracy at 50% IoU overlap threshold")
    print("  Higher is better (1.0 = perfect)")


def run_inference_on_sample():
    """
    Run the trained model on a few sample frames and save
    annotated images to OUTPUT_DIR/sample_predictions/
    """
    from ultralytics import YOLO
    import cv2

    best_model_path = Path(OUTPUT_DIR) / "runs" / "baseball_detect" / "weights" / "best.pt"
    if not best_model_path.exists():
        print("No trained model found.")
        return

    model    = YOLO(str(best_model_path))
    val_dir  = Path(OUTPUT_DIR) / "images" / "val"
    out_dir  = Path(OUTPUT_DIR) / "sample_predictions"
    out_dir.mkdir(exist_ok=True)

    val_images = list(val_dir.glob("*.jpg"))[:10]  # first 10 val images

    for img_path in val_images:
        results = model(str(img_path), imgsz=IMG_SIZE, conf=0.25)
        annotated = results[0].plot()
        cv2.imwrite(str(out_dir / img_path.name), annotated)

    print(f"\nSample predictions saved to: {out_dir}")
    print("Open that folder to visually inspect the model's detections.")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("Baseball Detection - Training Pipeline")
    print("=" * 50)

    # Step 1: Parse CVAT annotations
    print("\n[1/5] Parsing CVAT annotations...")
    annotations = parse_cvat_xml(CVAT_XML)

    if len(annotations) == 0:
        print("ERROR: No annotated frames found. Check your XML path and FRAMES_DIR.")
        exit(1)

    # Step 2: Build YOLO dataset
    print("\n[2/5] Building YOLO dataset...")
    n_train, n_val = build_yolo_dataset(annotations)

    # Step 3: Write YAML config
    print("\n[3/5] Writing dataset config...")
    yaml_path = write_yaml()

    # Step 4: Train
    print("\n[4/5] Training model...")
    train_model(yaml_path)

    # Step 5: Evaluate
    print("\n[5/5] Evaluating model...")
    evaluate_model()

    # Bonus: Save sample predictions
    print("\n[Bonus] Running inference on sample frames...")
    run_inference_on_sample()

    print("\nDone! Check OUTPUT_DIR for all results:")
    print(f"  {OUTPUT_DIR}")
