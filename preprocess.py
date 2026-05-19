# src/preprocess.py
import cv2
import os
from tqdm import tqdm

# Settings
INPUT_ROOT = "../data/images"           # original images folder structure: train/ val/ test/
OUTPUT_ROOT = "../data/images_preprocessed"
TARGET_SIZE = (640, 640)                # YOLO training size
APPLY_CLAHE = True
APPLY_DENOISE = True

def apply_clahe_color(img):
    """Apply CLAHE on each channel in LAB space for better contrast."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    final = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return final

def preprocess_image(img_path, out_path):
    img = cv2.imread(img_path)
    if img is None:
        return False
    # optional denoise Gaussian Filter
    if APPLY_DENOISE:
        img = cv2.GaussianBlur(img, (3,3), 0)
    # CLAHE
    if APPLY_CLAHE:
        img = apply_clahe_color(img)
    # resize (stretches to target). This keeps YOLO normalized labels valid.
    img = cv2.resize(img, TARGET_SIZE)
    # save
    cv2.imwrite(out_path, img)
    return True

def process_split(split_name):
    src_dir = os.path.join(INPUT_ROOT, split_name)
    dst_dir = os.path.join(OUTPUT_ROOT, split_name)
    os.makedirs(dst_dir, exist_ok=True)
    files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    for fn in tqdm(sorted(files), desc=f"Processing {split_name}"):
        in_path = os.path.join(src_dir, fn)
        out_path = os.path.join(dst_dir, fn)
        preprocess_image(in_path, out_path)

if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        if os.path.exists(os.path.join(INPUT_ROOT, split)):
            process_split(split)
        else:
            print(f"Skip {split}: folder not found at {os.path.join(INPUT_ROOT, split)}")
    print("Preprocessing finished. Processed images are at:", os.path.abspath(os.path.join(OUTPUT_ROOT)))
