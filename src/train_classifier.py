import os
import argparse
import time
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision
from torchvision import datasets, transforms
import timm
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Train Road Condition Classifier')
    parser.add_argument('--data-dir', type=str, default='dataset/classification/training', help='Path to dataset directory')
    parser.add_argument('--model', type=str, default='resnet18', 
                        choices=[
                            'resnet18', 'resnet34', 
                            'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
                            'vit_base_patch16_224', 'vit_small_patch16_224'
                        ], help='Model architecture')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--merge-classes', action='store_true', 
                        help='Merge Excellent/Good into Good, Fair/Poor into Bad for binary classification')
    parser.add_argument('--use-class-weights', action='store_true', 
                        help='Use weighted CrossEntropyLoss to handle class imbalance')
    return parser.parse_args()

class MergedDataset(torch.utils.data.Dataset):
    """Wrapper to merge specific classes dynamically"""
    def __init__(self, dataset, merge_mapping):
        self.dataset = dataset
        self.merge_mapping = merge_mapping
        # Map original class indices to new class indices
        self.classes = list(set(merge_mapping.values()))
        self.classes.sort()
        
        self.idx_to_new_idx = {}
        for old_idx, old_class in enumerate(dataset.classes):
            new_class = merge_mapping.get(old_class, old_class)
            new_idx = self.classes.index(new_class)
            self.idx_to_new_idx[old_idx] = new_idx
            
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        img, old_label = self.dataset[idx]
        new_label = self.idx_to_new_idx[old_label]
        return img, new_label

def get_model(model_name, num_classes):
    print(f"Loading {model_name} for {num_classes} classes...")
    # Use timm for easy loading of all these architectures with pre-trained weights
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    return model

def main():
    args = parse_args()
    
    # Check for ROCm GPU (AMD) / CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)} (ROCm/CUDA)")
    else:
        print("WARNING: CUDA/ROCm not found, using CPU. Training will be slow.")

    # Image transformations
    # ViT usually expects 224x224
    img_size = 224
    
    data_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load dataset
    print(f"Loading dataset from {args.data_dir}...")
    dataset = datasets.ImageFolder(args.data_dir, transform=data_transforms)
    
    classes = dataset.classes
    num_classes = len(classes)
    
    print(f"Original Classes found: {classes}")

    # Handle Class Merging
    if args.merge_classes:
        print("Merging classes: Excellent/Good -> Good, Fair/Poor -> Bad")
        merge_mapping = {
            'Excellent': 'Good',
            'Good': 'Good',
            'Fair': 'Bad',
            'Poor': 'Bad'
        }
        dataset = MergedDataset(dataset, merge_mapping)
        classes = dataset.classes
        num_classes = len(classes)
        print(f"New Merged Classes: {classes}")
        
        # Calculate new label distributions for weights
        labels = [dataset[i][1] for i in range(len(dataset))]
    else:
        labels = [label for _, label in dataset.imgs]

    # Calculate class weights for imbalance
    class_counts = Counter(labels)
    print("\nClass Distribution:")
    for i, c in enumerate(classes):
        print(f"  {c}: {class_counts[i]} images")
        
    class_weights = None
    if args.use_class_weights:
        # Inverse class frequency
        total_samples = sum(class_counts.values())
        weights = [total_samples / class_counts[i] for i in range(num_classes)]
        class_weights = torch.FloatTensor(weights).to(device)
        print(f"Computed Class Weights: {weights}")
        
    # DataLoader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Initialize Model
    model = get_model(args.model, num_classes)
    model = model.to(device)

    # Loss and Optimizer
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
        
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training Loop
    print(f"\nStarting training {args.model} for {args.epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            progress_bar.set_postfix({'loss': loss.item(), 'acc': 100.*correct/total})
            
        epoch_loss = running_loss / len(dataset)
        epoch_acc = 100. * correct / total
        print(f"Epoch {epoch+1} Summary: Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")
        
    time_elapsed = time.time() - start_time
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    # Save model
    os.makedirs('models', exist_ok=True)
    save_path = f"models/{args.model}_epochs{args.epochs}_acc{epoch_acc:.0f}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == '__main__':
    main()
