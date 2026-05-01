# Automated Baseball Detection in Training Videos
**Econ 8310 — Business Forecasting | Semester Project | April 2026**  
**Jungmin Hwang & Steph Simmons**

---

## Overview

This project develops a computer vision model that automatically detects moving baseballs in training videos and places tight bounding boxes around the ball in each frame. The model is built on YOLOv8 (You Only Look Once, version 8) using transfer learning, fine-tuned on annotated baseball training footage.

The repository contains the full pipeline: frame extraction, dataset construction, learning curve experiment (5 models trained at 10–100% of data), and a final best-effort model trained for 50 epochs on 100% of available data.

---

## Repository Structure

```
econ8310-baseball-detection/
├── yolo_learning_curve.ipynb   # Main script — run this
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── results/
    ├── learning_curve.png          # 5-fraction learning curve plot
    ├── learning_curve_results.csv  # Learning curve metrics (CSV)
    ├── results_50epoch.png         # Best model training plot (50 epochs)
    └── results_168frames.png       # Baseline model training plot (150 epochs)
```

---

## How to Run

### Prerequisites
- Python 3.10+
- Google Colab (recommended) or a machine with a CUDA-capable GPU
- Annotated video data from the course OneDrive

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```
Or in Colab:
```python
!pip install ultralytics -q
```

### Step 2 — Upload your data
If running on Google Colab, upload `moving_frames.zip` and `moving_labels.zip` to Google Drive, then mount Drive and unzip:
```python
from google.colab import drive
drive.mount('/content/drive')

import zipfile
with zipfile.ZipFile('/content/drive/MyDrive/moving_frames.zip', 'r') as z:
    z.extractall('/content/moving_data')
with zipfile.ZipFile('/content/drive/MyDrive/moving_labels.zip', 'r') as z:
    z.extractall('/content/moving_data')
```

### Step 3 — Update paths
At the top of `yolo_learning_curve.ipynb`, update only these three variables:
```python
XML_DIR    = "/path/to/Annotations"
VIDEO_DIR  = "/path/to/Raw Videos"
OUTPUT_DIR = "/path/to/Output"
```
If using pre-extracted frames (as in this repo), set all three to the folder containing `moving_frames/` and `moving_labels/`.

### Step 4 — Run the notebook
Run all cells top to bottom. The notebook will:
1. Load pre-extracted moving-ball frames and labels
2. Split into 80% train / 20% validation
3. Run a learning curve sweep (5 models at 10%, 30%, 50%, 70%, 100%)
4. Save results to `learning_curve_results.csv` and `learning_curve.png`

---

## Key Settings

| Parameter | Value | Description |
|---|---|---|
| `MODEL_SIZE` | `yolov8s.pt` | YOLOv8 small architecture |
| `EPOCHS` | `50` | Training epochs for best model |
| `TRAIN_FRACTIONS` | `[0.10, 0.30, 0.50, 0.70, 1.00]` | Learning curve fractions |
| `VAL_SPLIT` | `0.2` | 20% held out for validation |
| `RANDOM_SEED` | `42` | Fixed seed for reproducibility |
| `device` | `cuda` | Change to `mps` (Mac) or `cpu` |

---

## Results Summary

| Model | Frames | Epochs | mAP@0.5 | Precision | Recall |
|---|---|---|---|---|---|
| 168-frame baseline | 168 | 150 | 0.7929 | 0.9221 | 0.6961 |
| 5-epoch sweep (100%) | 490 | 5 | 0.2638 | 0.5956 | 0.2363 |
| **50-epoch best model** | **490** | **50** | **0.6119** | **0.8585** | **0.5666** |

---

## Dependencies

See `requirements.txt`. Key packages:
- `ultralytics` — YOLOv8
- `opencv-python` — frame extraction
- `pandas` — results logging
- `matplotlib` — learning curve plots
- `PyYAML` — dataset configuration

---

## Notes

- The `device` parameter auto-selects GPU if available. To manually set: `"cuda"` (NVIDIA), `"mps"` (Apple Silicon), or `"cpu"`.
- Two frames are flagged as corrupt and automatically excluded during training.
- The learning curve uses a fixed validation set across all 5 runs to ensure comparisons are directly meaningful.
