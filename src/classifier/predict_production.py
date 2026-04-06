#!/usr/bin/env python3
import os
import argparse
import random
import json
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
from tqdm import tqdm

# Alphabetical order as expected by PyTorch ImageFolder during training
CLASSES = ["Excellent", "Fair", "Good", "Invalid", "Poor"]

class InferenceDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            image = Image.open(path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, str(path)
        except Exception as e:
            # Return a dummy tensor and the path if image is corrupted
            return torch.zeros((3, 224, 224)), str(path)

def parse_args():
    parser = argparse.ArgumentParser(description='Classify Production Data')
    parser.add_argument('--data-dir', type=str, default='processed_data/filtered_frames', help='Path to production data directory')
    parser.add_argument('--model', type=str, default='vit_small_patch16_224', help='Model architecture')
    parser.add_argument('--weights', type=str, default='models/vit_small_patch16_224_epochs100_acc100.pth', help='Path to downloaded model weights')
    parser.add_argument('--output', type=str, default='production_predictions.json', help='Output JSON for all predictions')
    parser.add_argument('--sample-size', type=int, default=50, help='Number of images per class to sample for web UI')
    parser.add_argument('--eval-dir', type=str, default='road_classifier/eval_images', help='Directory to copy the samples to for app.py')
    parser.add_argument('--eval-output', type=str, default='eval_predictions.json', help='Output JSON for just the sampled evaluation dataset')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for inference')
    return parser.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(args.weights):
        print(f"Error: Weights file not found: {args.weights}")
        print("Please ensure you have downloaded the trained model weights from your server.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Model
    print(f"Loading {args.model}...")
    model = timm.create_model(args.model, pretrained=False, num_classes=len(CLASSES))
    
    # Load weights
    try:
        model.load_state_dict(torch.load(args.weights, map_location=device))
    except Exception as e:
        print(f"Failed to load weights: {e}")
        return

    model = model.to(device)
    model.eval()

    # Transformations matching train_classifier validation
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Find total images
    print(f"Scanning for images in {args.data_dir}...")
    image_paths = []
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    for root, _, files in os.walk(args.data_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in valid_exts:
                image_paths.append(os.path.join(root, f))
                
    if not image_paths:
        print(f"No images found in {args.data_dir}!")
        return
        
    print(f"Found {len(image_paths)} images.")

    dataset = InferenceDataset(image_paths, transform=data_transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Perform inference
    predictions = {}
    class_groups = {c: [] for c in CLASSES}

    print("Running inference...")
    with torch.no_grad():
        for inputs, paths in tqdm(dataloader, desc="Classifying"):
            inputs = inputs.to(device)
            # The dummy tensor check (black image) will output some class, but we can ignore errors
            outputs = model(inputs)
            _, preds = outputs.max(1)
            
            for i in range(len(paths)):
                predicted_class = CLASSES[preds[i].item()]
                path = paths[i]
                predictions[path] = predicted_class
                class_groups[predicted_class].append(path)

    # Save all production predictions
    with open(args.output, 'w') as f:
        json.dump(predictions, f, indent=4)
    print(f"Saved all predictions to {args.output}")

    # Print distribution
    print("\nProduction Data Distribution:")
    for c in CLASSES:
        print(f"  {c}: {len(class_groups[c])} images")

    # Sample for Web UI Evaluation
    print(f"\nSampling {args.sample_size} images per class for web evaluation...")
    os.makedirs(args.eval_dir, exist_ok=True)
    
    eval_predictions = {}
    total_sampled = 0
    
    for c in CLASSES:
        available_images = class_groups[c]
        if len(available_images) > args.sample_size:
            sampled = random.sample(available_images, args.sample_size)
        else:
            sampled = available_images
            
        for path in sampled:
            # Flatten name to avoid subdirectory conflicts in web UI
            # e.g., video1/frame_001.jpg -> video1_frame_001.jpg
            # Safe relative path from data_dir:
            try:
                rel_path = os.path.relpath(path, args.data_dir)
                flat_name = rel_path.replace(os.sep, '_')
            except ValueError:
                flat_name = os.path.basename(path)
                
            dest_path = os.path.join(args.eval_dir, flat_name)
            shutil.copy2(path, dest_path)
            
            # Save mapping with the flat_name as key, matching what road_classifier will use
            eval_predictions[flat_name] = c
            total_sampled += 1

    with open(args.eval_output, 'w') as f:
        json.dump(eval_predictions, f, indent=4)
        
    print(f"Copied {total_sampled} images to {args.eval_dir}")
    print(f"Saved evaluation mappings to {args.eval_output}")
    print(f"\nDone! You can now verify these predictions by running:")
    print(f"  cd road_classifier && python app.py --unfiltered-dir eval_images")

if __name__ == '__main__':
    main()
