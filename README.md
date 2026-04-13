# Baseball Detection — How to Run

## What this does
Trains a YOLOv8 object detection model to find baseballs in video frames,
using your CVAT-labeled data. Outputs a trained model + evaluation metrics.

---

## Step 1 — Install dependencies (one time only)

Open **Command Prompt**, navigate to this folder, then run:

```
python setup.py
```

This installs PyTorch + YOLOv8 (ultralytics), OpenCV, and other libraries.
Takes 2–5 minutes. You only need to do this once.

---

## Step 2 — Check your paths in train.py

Open `train.py` in any text editor and verify these three lines near the top:

```python
CVAT_XML   = r"C:\Users\req88455\Downloads\simmonscvat\simmonscvat.xml"
FRAMES_DIR = r"C:\Users\req88455\Downloads\Videos"
OUTPUT_DIR = r"C:\Users\req88455\Downloads\baseball_yolo"
```

- `CVAT_XML`   → path to your exported CVAT XML file
- `FRAMES_DIR` → folder containing your 7 video frame subfolders (IMG_8226_jared, etc.)
- `OUTPUT_DIR` → where the dataset and trained model will be saved (created automatically)

---

## Step 3 — Run training

```
python train.py
```

The script will:
1. Parse your CVAT XML annotations
2. Convert them to YOLO format and split into train/val sets
3. Download a pre-trained YOLOv8-nano model (small, ~6MB)
4. Fine-tune it on your baseball frames for 50 epochs
5. Print evaluation metrics (mAP, Precision, Recall)
6. Save sample prediction images so you can visually check results

**Training time on CPU:** approximately 20–60 minutes depending on your machine.

---

## Step 4 — Check your results

After training, look in `OUTPUT_DIR`:

```
baseball_yolo/
├── images/train/          ← training frames
├── images/val/            ← validation frames
├── labels/train/          ← YOLO format labels
├── labels/val/
├── dataset.yaml           ← dataset config
├── runs/baseball_detect/
│   ├── weights/
│   │   ├── best.pt        ← YOUR TRAINED MODEL
│   │   └── last.pt
│   ├── results.png        ← training curves
│   ├── confusion_matrix.png
│   └── val_batch*.jpg     ← sample predictions
└── sample_predictions/    ← model output on val frames
```

Open `results.png` to see your training curves (loss going down = good).
Open `sample_predictions/` to visually inspect what the model detects.

---

## Key metrics to report in your paper

From the console output after training:

| Metric | What it means |
|--------|--------------|
| mAP@0.5 | Detection accuracy at 50% overlap threshold — main metric |
| Precision | Of all boxes predicted, how many were correct |
| Recall | Of all actual balls, how many were found |

Higher = better for all three. A mAP@0.5 above 0.5 is a solid result
for a small dataset like this.

---

## Troubleshooting

**"No annotated frames found"**
→ Check CVAT_XML path and make sure FRAMES_DIR contains your frame folders

**"ModuleNotFoundError"**
→ Run `python setup.py` again

**Training is very slow**
→ Reduce EPOCHS to 20 in train.py, or reduce IMG_SIZE to 416

**Out of memory**
→ Reduce batch size: change `batch=4` to `batch=2` in train.py
