# src/detect.py
from ultralytics import YOLO
import cv2
import argparse
import os
from src.ocr_utilsi import ocr_read


def pad_box(x1,y1,x2,y2,pad,w,h):
    x1 = max(0, int(x1-pad))
    y1 = max(0, int(y1-pad))
    x2 = min(w, int(x2+pad))
    y2 = min(h, int(y2+pad))
    return x1,y1,x2,y2

def main(weights="runs/detect/train/weights/best.pt", image_path="data/images_preprocessed/test/sample.jpg", save_out=True):
    model = YOLO(weights)
    img = cv2.imread(image_path)
    if img is None:
        print("Image not found:", image_path)
        return
    h,w = img.shape[:2]
    results = model(img)
    out = img.copy()
    pad = 5
    for r in results:
        for box in r.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            x1,y1,x2,y2 = pad_box(x1,y1,x2,y2,pad,w,h)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            text = ocr_read(crop)
            print("Detected:", text)
            cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(out, text, (x1, max(15,y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    if save_out:
        out_path = os.path.splitext(os.path.basename(image_path))[0] + "_out.jpg"
        cv2.imwrite(out_path, out)
        print("Saved:", out_path)
    cv2.imshow("out", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="../runs/detect/train/weights/best.pt")
    parser.add_argument("--image", type=str, default="../data/images_preprocessed/test/sample.jpg")
    args = parser.parse_args()
    main(weights=args.weights, image_path=args.image)
