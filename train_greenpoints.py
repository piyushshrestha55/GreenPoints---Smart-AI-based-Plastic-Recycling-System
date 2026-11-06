
from ultralytics import YOLO


def main():

    # Load pretrained YOLO26 Nano
    model = YOLO("yolo26n.pt")

    # Train using the RTX 4050
    results = model.train(
        data="datasets/greenpoints_v1/data.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        device=0,
        patience=20,
        workers=0,  # Avoid Windows multiprocessing issues
        project="runs/greenpoints",
        name="yolo26n_v1",
        exist_ok=True
    )

    print("GreenPoints training completed!")


if __name__ == "__main__":
    main()
