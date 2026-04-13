# config.py — Edit this file with YOUR local paths
# This file should NOT be committed to GitHub (it's in .gitignore)

XML_DIR    = r"C:\Users\req88455\Downloads\baseball_detection\XMLs"
FRAMES_DIR = r"C:\Users\req88455\Downloads\baseball_detection\frames"
OUTPUT_DIR = r"C:\Users\req88455\Downloads\baseball_yolo"


# TRAINING SETTINGS (same for everyone)

MODEL_SIZE  = "yolov8n.pt"  # nano = fastest on CPU
EPOCHS      = 50
IMG_SIZE    = 640
VAL_SPLIT   = 0.2
RANDOM_SEED = 42
