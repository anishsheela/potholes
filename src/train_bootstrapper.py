import os
import argparse
import shutil
from ultralytics import YOLO
import torch

def train_bootstrapper(data_yaml, epochs=30, imgsz=640, output_weights="models/bootstrapper.pt"):
    """
    Trains a baseline YOLOv8n model on a public dataset (like Roboflow) to 
    create an initial "Bootstrapper" model. This model will be used to 
    auto-generate pre-annotations in Label Studio.
    """
    if not os.path.exists(data_yaml):
        print(f"Error: Could not find dataset config at {data_yaml}")
        return
        
    print(f"--- Starting Bootstrapper Training on {data_yaml} ---")
    model = YOLO('yolov8n.pt') 
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using hardware acceleration: {device}")

    # Train the base model on this generic dataset
    model.train(
        data=data_yaml, 
        epochs=epochs, 
        imgsz=imgsz,
        project='runs/detect',
        name='bootstrapper_train',
        device=device,
        exist_ok=True # Overwrite previous runs
    )
    
    # Find the most recent best.pt in the runs directory since ultralytics pathing can be unpredictable
    import glob
    weight_files = glob.glob('runs/**/weights/best.pt', recursive=True)
    if weight_files:
        weight_files.sort(key=os.path.getmtime, reverse=True)
        yolo_best_weights = weight_files[0]
        
        os.makedirs(os.path.dirname(output_weights), exist_ok=True)
        shutil.copy2(yolo_best_weights, output_weights)
        print("\n--- Bootstrapper Training Complete! ---")
        print(f"Successfully saved your Bootstrapper Model to: {output_weights}")
    else:
        print("\n--- Training Finished, but could not locate best weights to copy! ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train initial Label Studio Bootstrapper model on a public Roboflow dataset.")
    parser.add_argument("--data", type=str, default="roboflow_dataset/data.yaml", help="Path to Roboflow data.yaml file")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--out", type=str, default="models/bootstrapper.pt", help="Where to save final weights")
    
    args = parser.parse_args()
    train_bootstrapper(args.data, args.epochs, 640, args.out)
