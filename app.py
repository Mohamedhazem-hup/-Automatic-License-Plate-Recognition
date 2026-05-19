import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import re
from PIL import Image

# ── Page config ──
st.set_page_config(page_title="ALPR System", page_icon="🚗", layout="centered")

st.title("🚗 Automatic License Plate Recognition")
st.markdown("ارفع صورة سيارة وهيقرأ اللوحة تلقائياً")

# ── Load model & OCR ──
MODEL_PATH = r"C:\cv project 11\src-20251213T235015Z-1-001\src\runs\detect\train2\weights\best.pt"

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

model = load_model()
reader = load_ocr()

def clean_plate_text(text):
    t = text.strip().upper()
    t = re.sub(r'[^A-Z0-9]', '', t)
    if any(ch.isdigit() for ch in t):
        t = t.replace('O', '0').replace('I', '1')
    return t

def ocr_read(crop_bgr):
    results = reader.readtext(crop_bgr, detail=0)
    if not results:
        return ""
    text = "".join(results)
    return clean_plate_text(text)

# ── Upload ──
uploaded = st.file_uploader("ارفع صورة", type=["jpg", "jpeg", "png", "jfif"])

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="الصورة الأصلية", use_column_width=True)

    with st.spinner("جاري التحليل..."):
        results = model(img)
        plate_text = ""
        annotated = img.copy()

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                crop = img[y1:y2, x1:x2]
                plate_text = ocr_read(crop)

                # رسم المربع
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(annotated, plate_text, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                break

    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="النتيجة", use_column_width=True)

    if plate_text:
        st.success(f"🔍 اللوحة: **{plate_text}**")
    else:
        st.warning("⚠️ مش لاقي لوحة في الصورة")