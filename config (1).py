# config.py — Edit this file with YOUR local paths
# Each person on the team sets their own paths here.
# This file should NOT be committed to GitHub (it's in .gitignore)

# ─────────────────────────────────────────
# STEPH'S PATHS (example — update yours below)
# ─────────────────────────────────────────

XML_DIR    = r"C:\Users\req88455\Downloads\baseball_detection\XMLs"
FRAMES_DIR = r"C:\Users\req88455\Downloads\baseball_detection\frames"
OUTPUT_DIR = r"C:\Users\req88455\Downloads\baseball_yolo"

# ─────────────────────────────────────────
# PARTNER'S PATHS (uncomment and fill in)
# ─────────────────────────────────────────

# XML_DIR    = r"C:\Users\PARTNER\Downloads\baseball_detection\XMLs"
# FRAMES_DIR = r"C:\Users\PARTNER\Downloads\baseball_detection\frames"
# OUTPUT_DIR = r"C:\Users\PARTNER\Downloads\baseball_yolo"

# ─────────────────────────────────────────
# TRAINING SETTINGS (same for everyone)
# ─────────────────────────────────────────

MODEL_SIZE  = "yolov8n.pt"  # nano = fastest on CPU
EPOCHS      = 50
IMG_SIZE    = 640
VAL_SPLIT   = 0.2
RANDOM_SEED = 42
