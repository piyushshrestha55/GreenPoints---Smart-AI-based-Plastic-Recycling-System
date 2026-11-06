from ultralytics import YOLO


def main():

    # Load our trained model
    model = YOLO(
        r"runs/detect/runs/greenpoints/yolo26n_v1/weights/best.pt"
    )

    # Test on unseen images
    results = model.val(
        data="datasets/greenpoints_v1/data.yaml",
        split="test",
        imgsz=640,
        batch=8,
        device=0,
        plots=True
    )

    print("\nGreenPoints test completed!")


if __name__ == "__main__":
    main()