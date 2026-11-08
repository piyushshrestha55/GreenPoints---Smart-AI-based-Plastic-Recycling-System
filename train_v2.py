import os
from ultralytics import YOLO

def main():
    # 1. LOAD EXISTING KNOWLEDGE
    # Instead of starting from scratch (yolo26n.pt), we load your existing best.pt.
    # This ensures the model remembers the 1,100 images it already studied.
    init_weights = "weights/best.pt" if os.path.exists("weights/best.pt") else r"runs/detect/runs/greenpoints/yolo26n_v1/weights/best.pt"
    model = YOLO(init_weights)

    # 2. FINE-TUNE WITH OPTIMIZED SETTINGS
    results = model.train(
        # The new dataset folder (contains the 1100 old + 37 new augmented images)
        data="datasets/greenpoints_v2/data.yaml", 
        
        # Max training cycles. 50 is plenty since it already knows the basics.
        epochs=50,
        
        # Early stopping: if the model doesn't improve for 15 epochs in a row, stop early.
        patience=15,
        
        # INCREASED BATCH SIZE: Feeds 32 images at a time instead of 8.
        # This forces your RTX 4050 (6GB VRAM) to work harder and process epochs much faster.
        batch=16,
        
        # Standardizes all images to 640x640 pixels (matches your Roboflow preprocessing).
        imgsz=640,
        
        # device=0 tells the script to use your primary NVIDIA GPU instead of the CPU.
        device=0,
        
        # Keep at 0 for Windows to prevent multi-threading crashes during data loading.
        # CHANGE TO 4 (Pre-fetches images in the background)
        workers=4,         
        
        # CHANGE TO "disk" (Caches processed images to your drive)
        cache="disk",        
        
        # CRITICAL: A very low learning rate (0.0001). 
        # This tells the AI to make only *tiny* adjustments for the new webcam lighting 
        # without accidentally "forgetting" the original 1100 images.
        lr0=0.0001,  
        
        # Saves this new training run in a separate folder so your old model isn't overwritten.
        
        name="greenpoints_v2_finetuned",
        
        # Confirms that we are building on top of pre-existing weights.
        pretrained=True
    )
    
    print("\nFine-tuning completed successfully! New best.pt saved to runs/detect/greenpoints_v2_finetuned/weights/")

if __name__ == "__main__":
    main()