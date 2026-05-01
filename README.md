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
├── learning_curve.png          # 5-fraction learning curve plot
├── learning_curve_results.csv  # Learning curve metrics (CSV)
├── results_50epoch.png         # Best model training plot (50 epochs)
└── results_168frames.png       # Baseline model training plot (150 epochs)
```

---

## Data

The training data (pre-extracted frames and labels) is hosted on Google Drive. Download both zip files before running the notebook:

- **Moving Frames:** https://drive.google.com/file/d/1_mQiaATB2yXy2BgIvmwlR2qi1KdiGyCR/view?usp=share_link
- **Moving Labels:** https://drive.google.com/file/d/1wANEyPCtExq1I2IgjJpptYaGpbdLTvwM/view?usp=share_link

---

## How to Run

### Option A — Google Colab (Recommended)

1. Upload both zip files to your Google Drive
2. Open `yolo_learning_curve.ipynb` in Colab
3. Set runtime to **T4 GPU** (Runtime → Change runtime type → T4 GPU)
4. Add this cell at the top and run it first:

```python
!pip install ultralytics -q

from google.colab import drive
drive.mount('/content/drive')

import zipfile
with zipfile.ZipFile('/content/drive/MyDrive/moving_frames.zip', 'r') as z:
    z.extractall('/content/moving_data')
with zipfile.ZipFile('/content/drive/MyDrive/moving_labels.zip', 'r') as z:
    z.extractall('/content/moving_data')
print("Done!")
```

5. Update the three path variables at the top of the notebook:

```python
XML_DIR    = "/content/moving_data"
VIDEO_DIR  = "/content/moving_data"
OUTPUT_DIR = "/content/moving_data"
```

6. Run all cells

### Option B — Local Machine

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download and unzip both data files from the Google Drive links above

3. Update the three path variables at the top of the notebook:
```python
XML_DIR    = "/path/to/Annotations"
VIDEO_DIR  = "/path/to/Raw Videos"
OUTPUT_DIR = "/path/to/Output"
```

4. Run all cells in `yolo_learning_curve.ipynb`

---

## Key Settings

| Parameter | Value | Description |
|---|---|---|
| `MODEL_SIZE` | `yolov8s.pt` | YOLOv8 small architecture |
| `EPOCHS` | `50` | Training epochs for best model |
| `TRAIN_FRACTIONS` | `[0.10, 0.30, 0.50, 0.70, 1.00]` | Learning curve fractions |
| `VAL_SPLIT` | `0.2` | 20% held out for validation |
| `RANDOM_SEED` | `42` | Fixed seed for reproducibility |
| `device` | auto-detected | Selects cuda / mps / cpu automatically |

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

- Device is auto-detected: CUDA (NVIDIA GPU) → MPS (Apple Silicon) → CPU
- Two frames are flagged as corrupt and automatically excluded during training
- The learning curve uses a fixed validation set across all 5 runs to ensure comparisons are directly meaningful
