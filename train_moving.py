"""
Baseball Detection - YOLOv8 Training Pipeline (Moving Balls Only)
Econ 8310 - Semester Project
Authors: Steph Simmons & Jungmin King

This version filters to ONLY frames where the ball is actively in flight
(moving=true in CVAT annotations), giving us ~975 training frames vs
the original 168. All other settings match the best run (yolov8s, 100 epochs).

Before running:
    1. pip install -r requirements.txt
    2. Edit config.py with your local paths
    3. python train_moving.py
"""

import os
import xml.etree.ElementTree as ET
import shutil
import random
import yaml
from pathlib import Path

# Load user-specific paths and settings from config.py
try:
    from config import (
        XML_DIR, FRAMES_DIR, OUTPUT_DIR,
        MODEL_SIZE, IMG_SIZE, VAL_SPLIT, RANDOM_SEED
    )
except ImportError:
    print("ERROR: config.py not found.")
    print("Copy config.py, fill in your local paths, and try again.")
    exit(1)

# Override output dir so we don't overwrite the original 168-frame results
OUTPUT_DIR = OUTPUT_DIR.rstrip("/").rstrip("\\") + "_975frames"

# Best settings from previous run
EPOCHS = 100

# ─────────────────────────────────────────────────────────────
# VIDEO NAME MAPPING
# Maps CVAT XML folder names -> your local file prefixes
# e.g. "IMG_8226_jared/frame_0001.jpg" -> "jared1_0001.jpg"
# ─────────────────────────────────────────────────────────────
NAME_MAP = {
    "IMG_8226_jared": "jared1",
    "IMG_8241_jared": "jared2",
    "IMG_8242_jared": "jared3",
    "IMG_8243_jared": "jared4",
    "IMG_8252_zach":  "zach1",
    "IMG_8255_zach":  "zach2",
    "IMG_8256_zach":  "zach3",
}


def find_image(img_name, frames_dir):
    """
    Try to find the image file locally using multiple strategies.
    CVAT XML stores names like: IMG_8226_jared/frame_0001.jpg
    But local files may be named: jared1_0001.jpg or frame_0001.jpg
    """
    frames_dir = Path(frames_dir)

    # Strategy 1: exact relative path as stored in XML
    candidate = frames_dir / img_name.replace("/", os.sep)
    if candidate.exists():
        return str(candidate)

    # Strategy 2: search all subdirectories by filename
    filename = Path(img_name).name
    for match in frames_dir.rglob(filename):
        return str(match)

    # Strategy 3: use NAME_MAP to convert CVAT names to local names
    parts = img_name.replace("\\", "/").split("/")
    if len(parts) == 2:
        folder, fname = parts
        prefix = NAME_MAP.get(folder)
        if prefix:
            number = Path(fname).stem.replace("frame_", "")
            mapped = f"{prefix}_{number}.jpg"
            candidate = frames_dir / mapped
            if candidate.exists():
                return str(candidate)

    # Strategy 4: stem-only match
    stem = Path(img_name).stem
    for match in frames_dir.rglob(f"{stem}.*"):
        return str(match)

    return None


def parse_moving_only(xml_dir, frames_dir):
    """
    Parse every XML in xml_dir.
    ONLY returns frames that have at least one moving=true bounding box.
    This is the key difference from the original train.py — we filter
    to only balls in flight, ignoring stationary balls entirely.
    """
    xml_files = list(Path(xml_dir).glob("*.xml"))
    print(f"Found {len(xml_files)} XML files")

    all_annotations = []
    total_boxes    = 0
    skipped_no_img = 0
    skipped_no_mov = 0

    for xml_path in xml_files:
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except ET.ParseError as e:
            print(f"  WARNING: Could not parse {xml_path.name}: {e}")
            continue

        # Group moving boxes by frame index
        frame_boxes = {}

        for track in root.findall("track"):
            for box in track.findall("box"):
                # Skip interpolated/outside frames
                if int(box.attrib.get("outside", 0)) == 1:
                    continue

                # Only include boxes where moving=true
                attr = box.find("attribute[@name='moving']")
                is_moving = (
                    attr is not None
                    and attr.text is not None
                    and attr.text.strip().lower() == "true"
                )
                if not is_moving:
                    continue

                frame_idx = int(box.attrib["frame"])
                try:
                    xtl = float(box.attrib["xtl"])
                    ytl = float(box.attrib["ytl"])
                    xbr = float(box.attrib["xbr"])
                    ybr = float(box.attrib["ybr"])
                    if xbr > xtl and ybr > ytl:
                        # Use image dimensions from the XML if available
                        img_w = int(box.attrib.get("width",  2160))
                        img_h = int(box.attrib.get("height", 3840))
                        frame_boxes.setdefault(frame_idx, []).append(
                            (xtl, ytl, xbr, ybr, img_w, img_h)
                        )
                except (KeyError, ValueError):
                    continue

        if not frame_boxes:
            skipped_no_mov += 1
            continue

        # Now try to find the image for each annotated frame
        # CVAT track-based XMLs use frame indices, not image names
        # We need to find frames by stem + frame number
        stem = xml_path.stem

        for frame_idx, boxes in frame_boxes.items():
            # Try common frame naming patterns
            img_path = None
            for pattern in [
                f"{stem}_frame{frame_idx:06d}.jpg",
                f"{stem}_frame{frame_idx:04d}.jpg",
                f"{stem}_{frame_idx:06d}.jpg",
                f"{stem}_{frame_idx:04d}.jpg",
            ]:
                candidate = Path(frames_dir).rglob(pattern)
                match = next(candidate, None)
                if match:
                    img_path = str(match)
                    break

            # Also try NAME_MAP prefix patterns
            if not img_path:
                prefix = NAME_MAP.get(stem)
                if prefix:
                    for pattern in [
                        f"{prefix}_{frame_idx:04d}.jpg",
                        f"{prefix}_{frame_idx:06d}.jpg",
                    ]:
                        candidate = list(Path(frames_dir).rglob(pattern))
                        if candidate:
                            img_path = str(candidate[0])
                            break

            if not img_path:
                skipped_no_img += 1
                continue

            # Get actual image dimensions
            try:
                import cv2
                img = cv2.imread(img_path)
                if img is not None:
                    img_h_actual, img_w_actual = img.shape[:2]
                    boxes = [(xtl, ytl, xbr, ybr, img_w_actual, img_h_actual)
                             for (xtl, ytl, xbr, ybr, _, _) in boxes]
            except Exception:
                pass  # Use dimensions from XML if cv2 fails

            all_annotations.append((f"{stem}/frame_{frame_idx:06d}", img_path, boxes))
            total_boxes += len(boxes)

    print(f"Matched {len(all_annotations)} moving-ball frames with local images")
    print(f"Total moving bounding boxes: {total_boxes}")
    print(f"Skipped (no local image):    {skipped_no_img}")
    print(f"Skipped (no moving boxes):   {skipped_no_mov}")
    return all_annotations


def cvat_box_to_yolo(xtl, ytl, xbr, ybr, img_w, img_h):
    """Convert CVAT absolute pixel coords to YOLO normalized format.
    Class 0 = baseball (in flight) — single class, same as original train.py
    """
    cx = max(0.0, min(1.0, ((xtl + xbr) / 2) / img_w))
    cy = max(0.0, min(1.0, ((ytl + ybr) / 2) / img_h))
    w  = max(0.0, min(1.0, (xbr - xtl) / img_w))
    h  = max(0.0, min(1.0, (ybr - ytl) / img_h))
    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def build_yolo_dataset(annotations):
    """
    Copy images and write YOLO label .txt files.
    Deduplicates frames that appear in multiple XMLs.
    """
    seen   = set()
    unique = []
    for ann in annotations:
        if ann[1] not in seen:
            seen.add(ann[1])
            unique.append(ann)

    if len(unique) < len(annotations):
        print(f"Removed {len(annotations) - len(unique)} duplicate frames")

    random.seed(RANDOM_SEED)
    random.shuffle(unique)

    n_val   = max(1, int(len(unique) * VAL_SPLIT))
    val_set = unique[:n_val]
    trn_set = unique[n_val:]

    for split, data in [("train", trn_set), ("val", val_set)]:
        img_dir = Path(OUTPUT_DIR) / "images" / split
        lbl_dir = Path(OUTPUT_DIR) / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_name, img_full, boxes in data:
            safe_name = img_name.replace("/", "_").replace("\\", "_")
            if not safe_name.lower().endswith(".jpg"):
                safe_name += ".jpg"
            stem = Path(safe_name).stem

            shutil.copy2(img_full, img_dir / safe_name)

            with open(lbl_dir / f"{stem}.txt", "w") as f:
                for (xtl, ytl, xbr, ybr, iw, ih) in boxes:
                    f.write(cvat_box_to_yolo(xtl, ytl, xbr, ybr, iw, ih) + "\n")

    print(f"Dataset ready: {len(trn_set)} train, {len(val_set)} val frames")
    return len(trn_set), len(val_set)


def write_yaml():
    """Write dataset YAML config for YOLOv8."""
    yaml_path = Path(OUTPUT_DIR) / "dataset.yaml"
    config = {
        "path"  : str(OUTPUT_DIR).replace("\\", "/"),
        "train" : "images/train",
        "val"   : "images/val",
        "nc"    : 1,
        "names" : ["baseball"]
    }
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"YAML written: {yaml_path}")
    return str(yaml_path)


def train_model(yaml_path):
    """Fine-tune YOLOv8s on moving-ball frames only."""
    import os
    from ultralytics import YOLO

    # Prevent MPS memory spikes on 8GB M2
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

    print(f"\nLoading pre-trained model: {MODEL_SIZE}")
    model = YOLO(MODEL_SIZE)
    print(f"Training for {EPOCHS} epochs on MPS (Apple Silicon)...\n")

    model.train(
        data      = yaml_path,
        epochs    = EPOCHS,
        imgsz     = IMG_SIZE,
        batch     = 2,
        device    = "mps",
        project   = str(Path(OUTPUT_DIR) / "runs"),
        name      = "baseball_detect",
        patience  = 20,

        # Data augmentation — same as best run
        hsv_h     = 0.015,
        hsv_s     = 0.7,
        hsv_v     = 0.4,
        fliplr    = 0.5,
        mosaic    = 1.0,
        degrees   = 10.0,
        translate = 0.1,
        scale     = 0.5,

        save      = True,
        plots     = True,
        verbose   = True,
    )


def evaluate_model():
    """Evaluate the best saved model and print metrics."""
    import glob
    from ultralytics import YOLO

    runs = sorted(glob.glob(str(Path(OUTPUT_DIR) / "runs" / "baseball_detect*")))
    if not runs:
        print("No trained model found.")
        return

    best = Path(runs[-1]) / "weights" / "best.pt"
    print(f"\nEvaluating: {best}")
    model   = YOLO(str(best))
    metrics = model.val(
        data   = str(Path(OUTPUT_DIR) / "dataset.yaml"),
        imgsz  = IMG_SIZE,
        device = "mps",
    )

    print("\n========== RESULTS (975 frames) ==========")
    print(f"mAP@0.5:       {metrics.box.map50:.4f}")
    print(f"mAP@0.5-0.95:  {metrics.box.map:.4f}")
    print(f"Precision:     {metrics.box.mp:.4f}")
    print(f"Recall:        {metrics.box.mr:.4f}")
    print("==========================================")
    print("\nCompare to original 168-frame run:")
    print("  mAP@0.5:   0.7929")
    print("  Precision: 0.9221")
    print("  Recall:    0.6961")


def save_sample_predictions():
    """Run trained model on val images and save annotated outputs."""
    import glob
    from ultralytics import YOLO
    import cv2

    runs = sorted(glob.glob(str(Path(OUTPUT_DIR) / "runs" / "baseball_detect*")))
    if not runs:
        return

    best    = Path(runs[-1]) / "weights" / "best.pt"
    model   = YOLO(str(best))
    val_dir = Path(OUTPUT_DIR) / "images" / "val"
    out_dir = Path(OUTPUT_DIR) / "sample_predictions"
    out_dir.mkdir(exist_ok=True)

    for img_path in list(val_dir.glob("*.jpg"))[:10]:
        results   = model(str(img_path), imgsz=IMG_SIZE, conf=0.25)
        annotated = results[0].plot()
        cv2.imwrite(str(out_dir / img_path.name), annotated)

    print(f"Sample predictions saved to: {out_dir}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("Baseball Detection - Moving Frames Only (975 frames)")
    print("=" * 55)
    print(f"Model:         {MODEL_SIZE}")
    print(f"Epochs:        {EPOCHS}")
    print(f"Image size:    {IMG_SIZE}")
    print(f"XML folder:    {XML_DIR}")
    print(f"Frames folder: {FRAMES_DIR}")
    print(f"Output folder: {OUTPUT_DIR}")

    print("\n[1/5] Parsing moving-ball annotations...")
    annotations = parse_moving_only(XML_DIR, FRAMES_DIR)

    if len(annotations) == 0:
        print("\nERROR: No moving-ball frames found.")
        print("Check that FRAMES_DIR contains your JPEG frame files.")
        exit(1)

    print("\n[2/5] Building YOLO dataset...")
    build_yolo_dataset(annotations)

    print("\n[3/5] Writing dataset config...")
    yaml_path = write_yaml()

    print("\n[4/5] Training model...")
    train_model(yaml_path)

    print("\n[5/5] Evaluating model...")
    evaluate_model()

    print("\n[Bonus] Saving sample predictions...")
    save_sample_predictions()

    print(f"\nAll done! Results saved to:\n  {OUTPUT_DIR}")
