import os
import torch
from ultralytics import YOLO


def main():
    # Load tracked trained model from weights/ or local training runs
    weights_path = (
        "weights/best.pt"
        if os.path.exists("weights/best.pt")
        else r"runs/detect/greenpoints_v2_finetuned-3/weights/best.pt"
    )
    if not os.path.exists(weights_path):
        weights_path = r"runs/detect/runs/greenpoints/yolo26n_v1/weights/best.pt"

    print(f"Loading model checkpoint: {weights_path}")
    model = YOLO(weights_path)

    # Automatically select the latest available test dataset
    data_yaml = (
        "datasets/greenpoints_v2/data.yaml"
        if os.path.exists("datasets/greenpoints_v2/data.yaml")
        else "datasets/greenpoints_v1/data.yaml"
    )
    device = 0 if torch.cuda.is_available() else "cpu"

    print(f"Evaluating on {data_yaml} (device={device})...")
    metrics = model.val(
        data=data_yaml,
        split="test",
        imgsz=640,
        batch=8,
        device=device,
        plots=True,
    )

    print("\nGreenPoints evaluation completed successfully!")
    return metrics


if __name__ == "__main__":
    main()