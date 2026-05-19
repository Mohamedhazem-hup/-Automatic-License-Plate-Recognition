# src/evaluate.py
import os
import cv2
from ultralytics import YOLO
from ocr_utilsi import ocr_read
import argparse
import pandas as pd


def levenshtein_distance(a, b):
    """
    Compute Levenshtein edit distance between 2 strings.
    This replaces the editdistance library.
    """
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]

    for i in range(len(a) + 1):
        dp[i][0] = i

    for j in range(len(b) + 1):
        dp[0][j] = j

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )
    return dp[-1][-1]


def compute_metrics(model_path, test_images_dir, gt_csv):
    """
    Evaluate the ALPR system.
    gt_csv must contain: filename,plate
    """
    model = YOLO(model_path)

    df = pd.read_csv(gt_csv)
    total = len(df)
    exact = 0
    cer_total = 0.0

    print("Evaluating on", total, "images...\n")

    for idx, row in df.iterrows():
        filename = str(row['filename']).strip()
        gt_raw = row.get('plate', '')
        if pd.isna(gt_raw):
            # skip rows without ground-truth plate
            continue
        gt = str(gt_raw).strip().upper()


        img_path = os.path.join(test_images_dir, filename)
        img = None

        # Important: ultralytics patches cv2.imread and may raise FileNotFoundError
        # instead of returning None. So we must only call imread if the file exists.
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
        else:
            # fallback: sometimes filenames in CSV don't include the correct prefix
            base = os.path.basename(str(filename))
            fallback_path = os.path.join(test_images_dir, base)
            if os.path.exists(fallback_path):
                img_path = fallback_path
                img = cv2.imread(img_path)


        if img is None:
            print("Image not found:", img_path)
            continue

        results = model(img)
        pred_text = ""

        # Take first detection only
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = img[y1:y2, x1:x2]
                pred_text = ocr_read(crop)
                break

        # Accuracy check
        if pred_text == gt:
            exact += 1

        # CER (character error rate)
        distance = levenshtein_distance(pred_text, gt)
        cer = distance / max(1, len(gt))
        cer_total += cer

        print(f"{filename} | GT: {gt} | Pred: {pred_text} | CER: {cer:.3f}")

    # Final results
    print("\n=== Final Evaluation Results ===")
    print("Exact Match Accuracy:", exact / total)
    print("Mean CER:", cer_total / total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='../runs/detect/train/weights/best.pt')
    parser.add_argument('--testdir', type=str, default='../data/images/test')
    parser.add_argument('--gt', type=str, default='../data/test_gt.csv')

    args = parser.parse_args()

    compute_metrics(args.model, args.testdir, args.gt)
