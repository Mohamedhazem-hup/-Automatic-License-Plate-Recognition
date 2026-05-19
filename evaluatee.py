from ultralytics import YOLO

model = YOLO(r"D:\Programming\College\Semester 5\Computer Vision\alpr_project\runs\detect\train4\weights\best.pt")

metrics = model.val(data=r"C:\cv project 11\data-20251215T033328Z-1-001\data.yaml")

print(metrics)