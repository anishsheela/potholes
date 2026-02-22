import os
import glob
import random
import shutil
import yaml
import argparse
from ultralytics import YOLO

def create_dataset_split(base_dir, split_ratio=0.8):
    """
    Takes nested annotated/batch_X/images/ and labels/ directories and 
    combines them into a single train/ and val/ split required by YOLO.
    """
    annotated_dir = os.path.join(base_dir, 'annotated')
    
    split_dir = os.path.join(base_dir, 'split')
    train_images = os.path.join(split_dir, 'images', 'train')
    val_images = os.path.join(split_dir, 'images', 'val')
    train_labels = os.path.join(split_dir, 'labels', 'train')
    val_labels = os.path.join(split_dir, 'labels', 'val')
    
    # Create directories
    for p in [train_images, val_images, train_labels, val_labels]:
        os.makedirs(p, exist_ok=True)
        
    all_images = glob.glob(os.path.join(annotated_dir, '**', 'images', '*.jpg'), recursive=True)
    if not all_images:
        print(f"Error: No completed images found in {annotated_dir}")
        return None
        
    print(f"Found {len(all_images)} total images across all annotated batches for training.")
    
    # Process labels and handle negative samples (backgrounds)
    valid_data = []
    pothole_count = 0
    background_count = 0
    
    for img_path in all_images:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # Path logic: dataset/annotated/batch_X/images/img.jpg
        # Label path: dataset/annotated/batch_X/labels/img.txt
        batch_dir = os.path.dirname(os.path.dirname(img_path))
        label_dir = os.path.join(batch_dir, 'labels')
        label_path = os.path.join(label_dir, f"{base_name}.txt")
        
        # Ensure labels folder exists if creating negative samples
        os.makedirs(label_dir, exist_ok=True)
        
        if os.path.exists(label_path):
            pothole_count += 1
        else:
            # Create an empty .txt file to tell YOLO "this image has 0 potholes"
            open(label_path, 'a').close()
            background_count += 1
            
        valid_data.append((img_path, label_path))
            
    print(f"Dataset prepared: {pothole_count} Pothole images / {background_count} Background images.")
    
    if pothole_count == 0:
        print("Error: No labeled bounding boxes found. Please ensure you exported from MakeSense in YOLO format.")
        return None

    # Shuffle and split
    random.shuffle(valid_data)
    split_idx = int(len(valid_data) * split_ratio)
    train_data = valid_data[:split_idx]
    val_data = valid_data[split_idx:]
    
    print(f"Splitting data: {len(train_data)} Train / {len(val_data)} Validation")
    
    # Copy files
    for src_img, src_label in train_data:
        shutil.copy2(src_img, os.path.join(train_images, os.path.basename(src_img)))
        shutil.copy2(src_label, os.path.join(train_labels, os.path.basename(src_label)))
        
    for src_img, src_label in val_data:
        shutil.copy2(src_img, os.path.join(val_images, os.path.basename(src_img)))
        shutil.copy2(src_label, os.path.join(val_labels, os.path.basename(src_label)))
        
    return split_dir

def create_yaml(split_dir, yaml_path):
    """
    Creates the pothole.yaml configuration file for YOLO training.
    """
    config = {
        'path': os.path.abspath(split_dir),
        'train': 'images/train',
        'val': 'images/val',
        'names': {
            0: 'pothole'
        }
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
        
    print(f"Created YOLO dataset configuration at {yaml_path}")

def train_model(yaml_path, epochs, imgsz):
    """
    Loads YOLOv8n and trains it on the custom dataset.
    """
    print("\n--- Starting YOLOv8 Training ---")
    print("Initializing YOLOv8 Nano model...")
    # Load a pre-trained Nano model
    model = YOLO('yolov8n.pt') 
    
    import torch
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Enforcing hardware acceleration: {device}")

    # Train the model
    results = model.train(
        data=yaml_path, 
        epochs=epochs, 
        imgsz=imgsz,
        project='runs/detect',
        name='train',
        device=device,
        exist_ok=True # Overwrite previous training runs in this folder for simplicity
    )
    
    print("\n--- Training Complete! ---")
    print(f"The best model weights are saved at: runs/detect/train/weights/best.pt")
    print("You can now use predict_video.py to test it out.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="dataset", help="Base dataset directory containing images/ and labels/")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training (default: 640)")
    
    args = parser.parse_args()
    
    split_dir = create_dataset_split(args.data_dir)
    if split_dir:
        yaml_path = os.path.join(args.data_dir, 'pothole.yaml')
        create_yaml(split_dir, yaml_path)
        train_model(yaml_path, args.epochs, args.imgsz)
