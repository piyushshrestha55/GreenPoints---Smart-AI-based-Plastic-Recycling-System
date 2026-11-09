import os
from ultralytics import YOLO

# 1. Load your newly fine-tuned model (prioritizes tracked weights/best.pt)
model_path = "weights/best.pt" if os.path.exists("weights/best.pt") else "runs/detect/greenpoints_v2_finetuned-3/weights/best.pt"
model = YOLO(model_path)

# 2. Test it on your webcam (source="1" or "0"). 
# If you want to test a saved image instead, change "1" to the image file path like "test_image.jpg"
print(f"Starting webcam with '{model_path}'... Press 'q' in the video window to stop.")
results = model.predict(source="1", show=True, conf=0.5)
