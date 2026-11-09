import os
import sys
from ultralytics import YOLO

# 1. Load your newly fine-tuned model (prioritizes tracked weights/best.pt)
model_path = "weights/best.pt" if os.path.exists("weights/best.pt") else "runs/detect/greenpoints_v2_finetuned-3/weights/best.pt"
model = YOLO(model_path)

# 2. Camera source / test input:
# - Can pass via command line: python predict_webcam.py 0
# - Defaults to "0" (built-in webcam) or "1" (external / DroidCam virtual camera)
source = sys.argv[1] if len(sys.argv) > 1 else "0"

print(f"Starting standalone YOLO predictor with '{model_path}' on source={source}...")
print("Usage: python predict_webcam.py [0 | 1 | http://<ip>:4747/video | image.jpg]")
print("Press 'q' in the video window to stop.")
_ = model.predict(source=source, show=True, conf=0.5)
