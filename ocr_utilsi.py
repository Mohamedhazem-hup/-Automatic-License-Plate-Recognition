# src/ocr_utils.py
import easyocr
import re

# If your system has GPU & torch configured, set gpu=True.
reader = easyocr.Reader(['en'], gpu=False)

def clean_plate_text(text):
    # uppercase, remove odd chars, merge spaces
    t = text.strip().upper()
    t = re.sub(r'[^A-Z0-9]', '', t)
    # common fixes: O -> 0 when digits likely
    if any(ch.isdigit() for ch in t):
        t = t.replace('O', '0').replace('I', '1')
    return t

def ocr_read(crop_bgr):
    # EasyOCR expects BGR or RGB; pass the crop directly
    results = reader.readtext(crop_bgr, detail=0)
    if not results:
        return ""
    text = "".join(results)  # join parts
    return clean_plate_text(text)
