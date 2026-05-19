# src/train.py
from ultralytics import YOLO
import argparse
import random
import numpy as np
import torch
import os

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main(data_yaml="../data.yaml", epochs=30, imgsz=640, batch=8, model="yolov8n.pt"):
    set_seed(42)
    os.makedirs("runs", exist_ok=True)
    print("Training YOLOv8. This may take a while.")
    model = YOLO(model)
    model.train(data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default="../data.yaml")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--model', type=str, default='yolov8n.pt')
    args = parser.parse_args()
    main(data_yaml=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, model=args.model)
