#!/usr/bin/env python3
import os
import argparse
import shutil
import json

import torch
import torch.nn.functional as F
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
            return torch.zeros((3, 224, 224)), str(path)

def parse_args():
    parser = argparse.ArgumentParser(description='Active Learning - Mine Hard Examples')
    parser.add_argument('--data-dir', type=str, default='processed_data/filtered_frames', help='Path to production data directory')
    parser.add_argument('--model', type=str, default='vit_small_patch16_224', help='Model architecture')
    parser.add_argument('--weights', type=str, default='models/vit_small_patch16_224_epochs100_acc100.pth', help='Path to downloaded model weights')
    parser.add_argument('--sample-size', type=int, default=200, help='Total number of hard images to sample for web UI')
    parser.add_argument('--eval-dir', type=str, default='road_classifier/active_learning_images', help='Directory to copy the samples to for app.py')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for inference')
    return parser.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(args.weights):
        print(f"Error: Weights file not found: {args.weights}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"Loading {args.model}...")
    model = timm.create_model(args.model, pretrained=False, num_classes=len(CLASSES))
    
    try:
        model.load_state_dict(torch.load(args.weights, map_location=device))
    except Exception as e:
        print(f"Failed to load weights: {e}")
        return

    model = model.to(device)
    model.eval()

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

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

    # Store tuples of (path, max_probability, predicted_class)
    results = []

    print("Running inference to find model uncertainties...")
    with torch.no_grad():
        for inputs, paths in tqdm(dataloader, desc="Analyzing"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # Get probabilities using softmax
            probs = F.softmax(outputs, dim=1)
            max_probs, preds = probs.max(1)
            
            for i in range(len(paths)):
                predicted_class = CLASSES[preds[i].item()]
                confidence = max_probs[i].item()
                path = paths[i]
                
                results.append({
                    "path": path,
                    "confidence": confidence,
                    "predicted_class": predicted_class,
                    "all_probs": probs[i].cpu().numpy().tolist()
                })

    # Sort primarily by lowest confidence (hardest examples)
    # The model is most confused when probability is close to 1/num_classes (e.g. 0.25)
    results.sort(key=lambda x: x["confidence"])

    # Extract the hardest N images
    hard_samples = results[:args.sample_size]

    print(f"\nFound {len(hard_samples)} extremely challenging images.")
    if len(hard_samples) > 0:
        print(f"Lowest confidence: {hard_samples[0]['confidence']:.4f}")
        print(f"Highest confidence in this batch: {hard_samples[-1]['confidence']:.4f}")

    # Copy to eval_dir
    os.makedirs(args.eval_dir, exist_ok=True)
    
    total_copied = 0
    metadata = {}
    
    for item in tqdm(hard_samples, desc="Copying images"):
        path = item["path"]
        
        # Flatten name
        try:
            rel_path = os.path.relpath(path, args.data_dir)
            flat_name = rel_path.replace(os.sep, '_')
        except ValueError:
            flat_name = os.path.basename(path)
            
        dest_path = os.path.join(args.eval_dir, flat_name)
        shutil.copy2(path, dest_path)
        
        metadata[flat_name] = {
            "predicted_class": item["predicted_class"],
            "model_confidence": item["confidence"],
            "probabilities": {CLASSES[i]: item["all_probs"][i] for i in range(4)}
        }
        total_copied += 1

    metadata_path = os.path.join(args.eval_dir, 'active_learning_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"\nCopied {total_copied} hard images to {args.eval_dir}")
    print(f"Saved metadata to {metadata_path}")
    print(f"\nReady for manual review! Run exactly:")
    print(f"  cd road_classifier && python app.py --unfiltered-dir active_learning_images")

if __name__ == '__main__':
    main()
