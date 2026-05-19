🚗 Automatic License Plate Recognition (ALPR)
📌 Description

This project is an Automatic License Plate Recognition (ALPR) system that detects vehicles and extracts license plate numbers from images and videos using Deep Learning and Computer Vision techniques.

The system uses a trained object detection model (such as YOLO) to locate license plates, followed by OCR (Optical Character Recognition) to extract the text from the detected plates.

⚙️ Features
🚗 Detects vehicles and license plates in real-time
🔍 High-accuracy object detection using YOLO
🧠 OCR-based text extraction from plates
📸 Works on images and video streams
💾 Outputs recognized plate numbers in structured format
🧠 Technologies Used
Python
OpenCV
YOLO (Ultralytics)
EasyOCR / Tesseract
Deep Learning
📊 Workflow
Input image/video
Detect license plate using YOLO
Crop detected plate region
Apply OCR to extract text
Display/store result
🚀 Future Improvements
Improve accuracy in low-light conditions
Support multiple countries plate formats
Real-time deployment with web interface (Streamlit / Flask)
👨‍💻 Author
OCR Image
   ↓
Extract Text
   ↓
Merge Parts
   ↓
Remove Symbols
   ↓
Fix OCR Errors
   ↓
Return Plate Number
<img width="317" height="688" alt="Screenshot 2026-05-18 091353" src="https://github.com/user-attachments/assets/2fb25dcc-73ec-43d0-808f-4f4554442fa7" />

Developed by Mohammed Hazem
