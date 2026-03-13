import os
import glob
import random
import shutil
import yaml
import argparse
from ultralytics import YOLO

def create_dataset_split(base_dir, split_ratio=0.8, bg_ratio=0.3, reuse_split=False):
    """
    Takes nested annotated/batch_X/images/ and labels/ directories and 
    combines them into a single train/ and val/ split required by YOLO.
    """
    training_dir = os.path.join(base_dir, 'training')
    
    split_dir = os.path.join(base_dir, 'split')
    
    if reuse_split and os.path.exists(split_dir):
        if os.path.exists(os.path.join(split_dir, 'images', 'train')):
            print(f"Reusing existing dataset split at {split_dir}")
            return split_dir
            
    # Clear out the old split directory to prevent data leakage across runs
    if os.path.exists(split_dir):
        shutil.rmtree(split_dir)
        
    train_images = os.path.join(split_dir, 'images', 'train')
    val_images = os.path.join(split_dir, 'images', 'val')
    train_labels = os.path.join(split_dir, 'labels', 'train')
    val_labels = os.path.join(split_dir, 'labels', 'val')
    
    # Create directories
    for p in [train_images, val_images, train_labels, val_labels]:
        os.makedirs(p, exist_ok=True)
        
    all_images = glob.glob(os.path.join(training_dir, 'images', '*.jpg'))
    if not all_images:
        print(f"Error: No completed images found in {training_dir}")
        return None
        
    print(f"Found {len(all_images)} total images in {training_dir} for training.")
    
    # Process labels and handle negative samples (backgrounds)
    pothole_data = []
    background_data = []
    
    for img_path in all_images:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # Path logic: dataset/training/images/img.jpg
        # Label path: dataset/training/labels/img.txt
        label_dir = os.path.join(training_dir, 'labels')
        label_path = os.path.join(label_dir, f"{base_name}.txt")
        
        # Ensure labels folder exists if creating negative samples
        os.makedirs(label_dir, exist_ok=True)
        
        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            pothole_data.append((img_path, label_path))
        else:
            # Create an empty .txt file to tell YOLO "this image has 0 potholes"
            if not os.path.exists(label_path):
                open(label_path, 'a').close()
            background_data.append((img_path, label_path))
            
    pothole_count = len(pothole_data)
    background_count = len(background_data)
    
    # BALANCE: limit backgrounds to a fraction of the pothole images to prevent class imbalance
    max_backgrounds = int(pothole_count * bg_ratio)
    if background_count > max_backgrounds:
        random.shuffle(background_data)
        background_data = background_data[:max_backgrounds]
        print(f"⚠️  Reduced backgrounds from {background_count} to {len(background_data)} for balance")
        
    valid_data = pothole_data + background_data
            
    print(f"Balanced dataset: {len(pothole_data)} Pothole images / {len(background_data)} Background images.")
    
    if pothole_count == 0:
        print("Error: No labeled bounding boxes found. Please ensure you exported from Lablestudio in YOLO format.")
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

def train_model(yaml_path, epochs, imgsz, base_weights, save_path, run_name='train'):
    """
    Loads YOLO configuration and trains it on the custom dataset.
    """
    print("\n--- Starting YOLOv8 Training ---")
    print(f"Initializing YOLO model from weights: {base_weights}...")
    # Load the requested pre-trained or custom model
    model = YOLO(base_weights) 
    
    import torch
    if torch.cuda.is_available():
        device = 'cuda:0'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Enforcing hardware acceleration: {device}")

    # Train the model
    results = model.train(
        data=yaml_path, 
        epochs=epochs, 
        imgsz=imgsz,
        project='runs/segment',
        name=run_name,
        device=device,
        exist_ok=True # Overwrite previous training runs in this folder for simplicity
    )
    
    # Dynamically locate the best.pt file since ultralytics file pathing can be unpredictable
    import glob
    weight_files = glob.glob('runs/**/weights/best.pt', recursive=True)
    if weight_files:
        weight_files.sort(key=os.path.getmtime, reverse=True)
        yolo_best_weights = weight_files[0]
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        shutil.copy2(yolo_best_weights, save_path)
        print("\n--- Training Complete! ---")
        print(f"Successfully copied the best model weights safely to: {save_path}")
        print("You can now use predict_video.py to test it out.")
        
        # Read and print final metrics from results.csv
        run_dir = os.path.dirname(os.path.dirname(yolo_best_weights))
        results_csv = os.path.join(run_dir, 'results.csv')
        if os.path.exists(results_csv):
            import csv
            try:
                with open(results_csv, 'r') as f:
                    reader = csv.reader(f)
                    headers = [h.strip() for h in next(reader)]
                    last_row = None
                    for row in reader:
                        if row: last_row = row
                    if last_row:
                        print("\n--- Final Training & Validation Statistics ---")
                        for h, v in zip(headers, last_row):
                            if h != 'epoch' and v.strip():
                                try:
                                    print(f"  {h}: {float(v):.4f}")
                                except ValueError:
                                    pass
            except Exception as e:
                print(f"Could not parse results.csv: {e}")
                
    else:
        print("\n--- Training Finished, but could not locate best weights to copy! ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="dataset", help="Base dataset directory containing images/ and labels/")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training (default: 640)")
    parser.add_argument("--model", type=str, default=None, help="Explicitly force init model (e.g., yolov8n.pt). Otherwise auto-resumes from models/best_pothole.pt")
    parser.add_argument("--bg-ratio", type=float, default=0.3, help="Maximum ratio of background images to pothole images (default: 0.3)")
    parser.add_argument("--run-name", type=str, default="train", help="Name of the run folder in runs/segment/")
    parser.add_argument("--save-weights", type=str, default="models/best_pothole.pt", help="Path to save the best weights")
    parser.add_argument("--reuse-split", action="store_true", help="Reuse existing dataset split if it exists")
    
    args = parser.parse_args()
    
    custom_weights_path = args.save_weights
    
    if args.model:
        weights_to_use = args.model
    elif os.path.exists(custom_weights_path):
        weights_to_use = custom_weights_path
        print(f"Found existing custom model at {custom_weights_path}. Will resume training from it.")
    else:
        weights_to_use = "yolov8n.pt"
        print(f"No existing custom model found. Starting fresh from base {weights_to_use}...")
    
    split_dir = create_dataset_split(args.data_dir, bg_ratio=args.bg_ratio, reuse_split=args.reuse_split)
    if split_dir:
        yaml_path = os.path.join(args.data_dir, 'pothole.yaml')
        create_yaml(split_dir, yaml_path)
        train_model(yaml_path, args.epochs, args.imgsz, weights_to_use, custom_weights_path, args.run_name)

