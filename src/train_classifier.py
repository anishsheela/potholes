import os
import argparse
import time
from collections import Counter
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision
from torchvision import datasets, transforms
import timm
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json

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
    parser.add_argument('--val-split', type=float, default=0.3, help='Validation split ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--include-invalid', action='store_true',
                        help='Include Invalid class in training (default: exclude)')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience (epochs)')
    return parser.parse_args()

class DatasetWithTransform(torch.utils.data.Dataset):
    """Wrapper to apply specific transform to a subset"""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __len__(self):
        return len(self.subset)
        
    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

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

class FilteredDataset(torch.utils.data.Dataset):
    """Dataset wrapper that filters out specific classes and remaps labels"""
    def __init__(self, dataset, valid_classes):
        self.dataset = dataset
        self.valid_classes = valid_classes
        self.old_to_new = {dataset.class_to_idx[c]: i for i, c in enumerate(valid_classes)}
        
        # Filter samples to only include valid classes
        self.samples = [
            (path, self.old_to_new[label]) 
            for path, label in dataset.samples 
            if dataset.classes[label] in valid_classes
        ]
        self.classes = valid_classes
        self.class_to_idx = {c: i for i, c in enumerate(valid_classes)}
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = self.dataset.loader(path)
        return img, label

def get_model(model_name, num_classes):
    print(f"Loading {model_name} for {num_classes} classes...")
    # Use timm for easy loading of all these architectures with pre-trained weights
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    return model

def main():
    args = parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Check for ROCm GPU (AMD) / CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)} (ROCm/CUDA)")
    else:
        print("WARNING: CUDA/ROCm not found, using CPU. Training will be slow.")

    # Image transformations
    # ViT usually expects 224x224
    img_size = 224
    
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load dataset WITHOUT transforms first to preserve PIL Images for custom subset
    print(f"Loading dataset from {args.data_dir}...")
    dataset_full = datasets.ImageFolder(args.data_dir, transform=None)
    
    classes = dataset_full.classes
    num_classes = len(classes)
    
    print(f"Original Classes found: {classes}")
    
    # Filter out Invalid class unless explicitly requested
    if 'Invalid' in classes and not args.include_invalid:
        print("Filtering out 'Invalid' class from training (use --include-invalid to keep it)...")
        valid_classes = [c for c in classes if c != 'Invalid']
        dataset_full = FilteredDataset(dataset_full, valid_classes)
        classes = valid_classes
        num_classes = len(classes)
        print(f"Training on {num_classes} classes: {classes}")

    # Handle Class Merging
    if args.merge_classes:
        print("Merging classes: Excellent/Good -> Good, Fair/Poor -> Bad")
        merge_mapping = {
            'Excellent': 'Good',
            'Good': 'Good',
            'Fair': 'Bad',
            'Poor': 'Bad'
        }
        dataset_full = MergedDataset(dataset_full, merge_mapping)
        classes = dataset_full.classes
        num_classes = len(classes)
        print(f"New Merged Classes: {classes}")
        
    # Get labels for split/weights
    if args.merge_classes:
        labels = [dataset_full[i][1] for i in range(len(dataset_full))]
    else:
        labels = [label for _, label in dataset_full.samples]

    # Calculate grouping for Train/Val split
    groups = []
    # If MergedDataset is used, the core ImageFolder is inside .dataset
    core_samples = dataset_full.dataset.samples if args.merge_classes else dataset_full.samples
    for path, _ in core_samples:
        filename = os.path.basename(path)
        if '_frame_' in filename:
            video_id = filename.split('_frame_')[0]
        else:
            video_id = filename
        groups.append(video_id)

    # Perform grouped split
    print(f"Splitting dataset with validation split {args.val_split} based on {len(set(groups))} unique videos...")
    gss = GroupShuffleSplit(n_splits=1, test_size=args.val_split, random_state=args.seed)
    
    try:
        train_idx, val_idx = next(gss.split(range(len(dataset_full)), labels, groups))
    except ValueError as e:
        print("Warning: GroupShuffleSplit failed (likely too few groups). Falling back to basic random split.")
        from sklearn.model_selection import train_test_split
        train_idx, val_idx = train_test_split(range(len(dataset_full)), test_size=args.val_split, random_state=args.seed, stratify=labels)

    train_subset = torch.utils.data.Subset(dataset_full, train_idx)
    val_subset = torch.utils.data.Subset(dataset_full, val_idx)

    train_dataset = DatasetWithTransform(train_subset, transform=train_transforms)
    val_dataset = DatasetWithTransform(val_subset, transform=val_transforms)

    print(f"Split Dataset: {len(train_dataset)} train images, {len(val_dataset)} validation images.")

    # Calculate class weights for imbalance (USING ONLY TRAINING DATA)
    train_labels = [labels[i] for i in train_idx]
    class_counts = Counter(train_labels)
    print("\nTraining Class Distribution:")
    for i, c in enumerate(classes):
        print(f"  {c}: {class_counts[i]} images")
        
    class_weights = None
    if args.use_class_weights:
        total_samples = sum(class_counts.values())
        weights = [total_samples / class_counts[i] for i in range(num_classes)]
        class_weights = torch.FloatTensor(weights).to(device)
        print(f"Computed Class Weights: {weights}")
        
    # DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

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
    print(f"\nStarting training {args.model} for max {args.epochs} epochs with patience {args.patience}...")
    start_time = time.time()
    
    best_val_loss = float('inf')
    patience_counter = 0
    final_epoch_acc = 0
    final_epoch_precision = 0
    final_epoch_recall = 0
    final_epoch_f1 = 0
    epochs_trained = 0
    
    os.makedirs('models', exist_ok=True)
    best_model_path = f"models/{args.model}_best.pth"

    for epoch in range(args.epochs):
        # TRAIN PHASE
        model.train()
        running_train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            progress_bar.set_postfix({'loss': loss.item(), 'acc': 100.*train_correct/train_total})
            
        epoch_train_loss = running_train_loss / len(train_dataset)
        epoch_train_acc = 100. * train_correct / train_total

        # VAL PHASE
        model.eval()
        running_val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_val_targets = []
        all_val_preds = []

        with torch.no_grad():
            for inputs, targets in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                running_val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
                all_val_targets.extend(targets.cpu().numpy())
                all_val_preds.extend(predicted.cpu().numpy())

        epoch_val_loss = running_val_loss / len(val_dataset)
        epoch_val_acc = 100. * val_correct / val_total
        
        # Calculate sklearn metrics (macro average for multiclass)
        epoch_precision = precision_score(all_val_targets, all_val_preds, average='macro', zero_division=0)
        epoch_recall = recall_score(all_val_targets, all_val_preds, average='macro', zero_division=0)
        epoch_f1 = f1_score(all_val_targets, all_val_preds, average='macro', zero_division=0)
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train -> Loss: {epoch_train_loss:.4f} | Acc: {epoch_train_acc:.2f}%")
        print(f"  Val   -> Loss: {epoch_val_loss:.4f} | Acc: {epoch_val_acc:.2f}% | P: {epoch_precision:.4f} | R: {epoch_recall:.4f} | F1: {epoch_f1:.4f}")
        
        epochs_trained += 1

        # EARLY STOPPING CHECK
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            final_epoch_acc = epoch_val_acc
            final_epoch_precision = epoch_precision
            final_epoch_recall = epoch_recall
            final_epoch_f1 = epoch_f1
            torch.save(model.state_dict(), best_model_path)
            print(f"  --> Saved new best model to {best_model_path} (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  --> Early stopping patience: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print("Triggered early stopping!")
                break
        
    time_elapsed = time.time() - start_time
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f"Best Validation Acc: {final_epoch_acc:.2f}%")

    # Output JSON payload at the very end for the tuner script to parse
    final_metrics = {
        "model": args.model,
        "epochs": epochs_trained,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "time_seconds": time_elapsed,
        "accuracy": final_epoch_acc,
        "precision": final_epoch_precision * 100,
        "recall": final_epoch_recall * 100,
        "f1_score": final_epoch_f1 * 100
    }
    print("\n--- JSON_METRICS_START ---")
    print(json.dumps(final_metrics))
    print("--- JSON_METRICS_END ---")

if __name__ == '__main__':
    main()
