# train.py - CORRECTED FOR WINDOWS
from ultralytics import YOLO
import torch

def main():
    """Main training function"""
    # Check GPU
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        device = 0  # Use GPU 0
    else:
        device = 'cpu'

    # Load model
    model = YOLO('yolov8m.pt')

    # Train with GPU optimizations
    model.train(
        data='helmet_dataset.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        workers=4,
        device=device,
        optimizer='auto',
        amp=True,
        cache=False,  # Changed from True to False to avoid RAM issues
        name='helmet_detection',
        exist_ok=True,
        patience=20,
        save=True,
        plots=True
    )

    print("✅ Training completed!")

# THIS IS THE CRITICAL LINE FOR WINDOWS
if __name__ == '__main__':
    # Optional: Set multiprocessing start method
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    main()