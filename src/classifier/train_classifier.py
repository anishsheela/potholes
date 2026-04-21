import os
import argparse
import time
from collections import Counter
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import timm
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import json


def parse_args():
    parser = argparse.ArgumentParser(description='Train Road Condition Classifier')
    parser.add_argument('--data-dir', type=str, default='dataset/classification/training')
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=[
                            'resnet18', 'resnet34', 'resnet50',
                            'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
                            'efficientnetv2_s', 'efficientnetv2_m',
                            'vit_base_patch16_224', 'vit_small_patch16_224',
                            'convnext_tiny', 'convnext_small', 'convnext_base',
                            'swin_tiny_patch4_window7_224', 'swin_small_patch4_window7_224'
                        ])
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--merge-classes', action='store_true',
                        help='Merge Excellent/Good -> Good, Fair/Poor -> Bad')
    parser.add_argument('--use-class-weights', action='store_true',
                        help='Use weighted CrossEntropyLoss to handle class imbalance')
    parser.add_argument('--val-split', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--include-invalid', action='store_true',
                        help='Include Invalid class in training (default: exclude)')
    parser.add_argument('--patience', type=int, default=3,
                        help='Early stopping patience (epochs)')
    # ── New improvements ──────────────────────────────────────────────────────
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                        help='Label smoothing for CrossEntropyLoss (0 = off)')
    parser.add_argument('--freeze-epochs', type=int, default=5,
                        help='Epochs to train head only before unfreezing backbone (0 = off)')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'plateau', 'none'],
                        help='LR scheduler for main training phase')
    return parser.parse_args()


# ── Dataset wrappers ──────────────────────────────────────────────────────────

class DatasetWithTransform(torch.utils.data.Dataset):
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
    def __init__(self, dataset, merge_mapping):
        self.dataset = dataset
        self.merge_mapping = merge_mapping
        self.classes = sorted(set(merge_mapping.values()))
        self.idx_to_new_idx = {}
        for old_idx, old_class in enumerate(dataset.classes):
            new_class = merge_mapping.get(old_class, old_class)
            self.idx_to_new_idx[old_idx] = self.classes.index(new_class)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, old_label = self.dataset[idx]
        return img, self.idx_to_new_idx[old_label]


class FilteredDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, valid_classes):
        self.dataset = dataset
        self.valid_classes = valid_classes
        self.old_to_new = {dataset.class_to_idx[c]: i for i, c in enumerate(valid_classes)}
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


# ── Model helpers ─────────────────────────────────────────────────────────────

def get_model(model_name, num_classes):
    print(f"Loading {model_name} for {num_classes} classes...")
    return timm.create_model(model_name, pretrained=True, num_classes=num_classes)


def freeze_backbone(model):
    """Freeze all layers except the classification head."""
    classifier = model.get_classifier()
    classifier_ids = {id(p) for p in classifier.parameters()}
    frozen = 0
    for p in model.parameters():
        if id(p) not in classifier_ids:
            p.requires_grad = False
            frozen += 1
    print(f"  Froze {frozen} parameter tensors. Training head only.")


def unfreeze_all(model):
    """Unfreeze every layer."""
    for p in model.parameters():
        p.requires_grad = True
    print(f"  Unfroze all parameters.")


# ── Training helpers ──────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, phase, epoch_label):
    is_train = (phase == 'train')
    model.train() if is_train else model.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    all_targets = []
    all_preds = []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        pbar = tqdm(loader, desc=epoch_label)
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            if is_train:
                optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            if is_train:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

            if is_train:
                pbar.set_postfix({'loss': f'{loss.item():.3f}',
                                  'acc': f'{100.*correct/total:.1f}%'})

    avg_loss = running_loss / total
    acc = 100. * correct / total
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall    = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1        = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    return avg_loss, acc, precision, recall, f1


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)} (ROCm/CUDA)")
    else:
        print("WARNING: CUDA/ROCm not found, using CPU. Training will be slow.")

    img_size = 224

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(8),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # ── Dataset ───────────────────────────────────────────────────────────────
    print(f"Loading dataset from {args.data_dir}...")
    dataset_full = datasets.ImageFolder(args.data_dir, transform=None)
    classes = dataset_full.classes
    print(f"Original Classes found: {classes}")

    if 'Invalid' in classes and not args.include_invalid:
        print("Filtering out 'Invalid' class (use --include-invalid to keep)...")
        valid_classes = [c for c in classes if c != 'Invalid']
        dataset_full = FilteredDataset(dataset_full, valid_classes)
        classes = valid_classes
        print(f"Training on {len(classes)} classes: {classes}")

    if args.merge_classes:
        print("Merging classes: Excellent/Good -> Good, Fair/Poor -> Bad")
        merge_mapping = {'Excellent': 'Good', 'Good': 'Good', 'Fair': 'Bad', 'Poor': 'Bad'}
        if args.include_invalid:
            merge_mapping['Invalid'] = 'Invalid'
        dataset_full = MergedDataset(dataset_full, merge_mapping)
        classes = dataset_full.classes
        print(f"Merged Classes: {classes}")

    num_classes = len(classes)

    labels = ([dataset_full[i][1] for i in range(len(dataset_full))]
              if args.merge_classes else
              [label for _, label in dataset_full.samples])

    # Group by video for split
    core_samples = dataset_full.dataset.samples if args.merge_classes else dataset_full.samples
    groups = []
    for path, _ in core_samples:
        filename = os.path.basename(path)
        groups.append(filename.split('_frame_')[0] if '_frame_' in filename else filename)

    print(f"Splitting dataset (val={args.val_split}) across {len(set(groups))} unique videos...")
    gss = GroupShuffleSplit(n_splits=1, test_size=args.val_split, random_state=args.seed)
    try:
        train_idx, val_idx = next(gss.split(range(len(dataset_full)), labels, groups))
    except ValueError:
        print("Warning: GroupShuffleSplit failed. Falling back to stratified split.")
        from sklearn.model_selection import train_test_split
        train_idx, val_idx = train_test_split(
            range(len(dataset_full)), test_size=args.val_split,
            random_state=args.seed, stratify=labels)

    train_dataset = DatasetWithTransform(
        torch.utils.data.Subset(dataset_full, train_idx), transform=train_transforms)
    val_dataset = DatasetWithTransform(
        torch.utils.data.Subset(dataset_full, val_idx), transform=val_transforms)

    print(f"Split: {len(train_dataset)} train / {len(val_dataset)} val")

    train_labels = [labels[i] for i in train_idx]
    class_counts = Counter(train_labels)
    print("\nTraining Class Distribution:")
    for i, c in enumerate(classes):
        print(f"  {c}: {class_counts[i]} images")

    class_weights = None
    if args.use_class_weights:
        total = sum(class_counts.values())
        weights = [total / class_counts[i] for i in range(num_classes)]
        class_weights = torch.FloatTensor(weights).to(device)
        print(f"Class Weights: {[f'{w:.2f}' for w in weights]}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4)

    # ── Model & loss ──────────────────────────────────────────────────────────
    model = get_model(args.model, num_classes).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing
    )
    if args.label_smoothing > 0:
        print(f"Label smoothing: {args.label_smoothing}")

    os.makedirs('models', exist_ok=True)
    best_model_path = f"models/{args.model}_best.pth"

    start_time = time.time()
    best_val_loss = float('inf')
    best_metrics  = {'acc': 0, 'precision': 0, 'recall': 0, 'f1': 0}
    total_epochs_trained = 0

    # ── Phase 1: Freeze backbone, warm up head ────────────────────────────────
    if args.freeze_epochs > 0:
        print(f"\n{'='*60}")
        print(f"Phase 1: Training head only for {args.freeze_epochs} epochs")
        print(f"{'='*60}")
        freeze_backbone(model)

        optimizer_p1 = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr
        )

        for epoch in range(args.freeze_epochs):
            label = f"Epoch {epoch+1}/{args.freeze_epochs} [Phase1-Train]"
            train_loss, train_acc, _, _, _ = run_epoch(
                model, train_loader, criterion, optimizer_p1, device, 'train', label)

            val_loss, val_acc, val_p, val_r, val_f1 = run_epoch(
                model, val_loader, criterion, optimizer_p1, device, 'val',
                f"Epoch {epoch+1}/{args.freeze_epochs} [Phase1-Val]")

            print(f"  Phase1 Epoch {epoch+1}: "
                  f"train_loss={train_loss:.4f} | "
                  f"val_loss={val_loss:.4f} | val_acc={val_acc:.2f}% | F1={val_f1:.4f}")

            total_epochs_trained += 1

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_metrics = {'acc': val_acc, 'precision': val_p,
                                'recall': val_r, 'f1': val_f1}
                torch.save(model.state_dict(), best_model_path)
                print(f"  --> Saved best model (val_loss={best_val_loss:.4f})")

        # Unfreeze for phase 2
        print(f"\n{'='*60}")
        print(f"Phase 2: Unfreezing all layers — fine-tuning at lr={args.lr/10:.2e}")
        print(f"{'='*60}")
        unfreeze_all(model)
    else:
        print(f"\nFreeze phase skipped (--freeze-epochs 0)")

    # ── Phase 2: Full fine-tuning with scheduler ──────────────────────────────
    phase2_lr = args.lr / 10 if args.freeze_epochs > 0 else args.lr
    optimizer = optim.Adam(model.parameters(), lr=phase2_lr)

    remaining_epochs = args.epochs - args.freeze_epochs
    if remaining_epochs <= 0:
        print("Warning: all epochs used in freeze phase. Increase --epochs.")
        remaining_epochs = 0

    # Set up scheduler
    scheduler = None
    if args.scheduler == 'cosine' and remaining_epochs > 0:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=remaining_epochs, eta_min=1e-7)
        print(f"Scheduler: CosineAnnealingLR over {remaining_epochs} epochs")
    elif args.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        print(f"Scheduler: ReduceLROnPlateau (factor=0.5, patience=3)")

    patience_counter = 0

    print(f"\nStarting {'phase 2 ' if args.freeze_epochs > 0 else ''}training for "
          f"up to {remaining_epochs} epochs (patience={args.patience})...")

    for epoch in range(remaining_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        label = f"Epoch {epoch+1}/{remaining_epochs} [Train] lr={current_lr:.2e}"

        train_loss, train_acc, _, _, _ = run_epoch(
            model, train_loader, criterion, optimizer, device, 'train', label)

        val_loss, val_acc, val_p, val_r, val_f1 = run_epoch(
            model, val_loader, criterion, optimizer, device, 'val',
            f"Epoch {epoch+1}/{remaining_epochs} [Val]")

        # Step scheduler
        if scheduler is not None:
            if args.scheduler == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()

        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train -> Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"  Val   -> Loss: {val_loss:.4f}  | Acc: {val_acc:.2f}% "
              f"| P: {val_p:.4f} | R: {val_r:.4f} | F1: {val_f1:.4f}")

        total_epochs_trained += 1

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_metrics = {'acc': val_acc, 'precision': val_p,
                            'recall': val_r, 'f1': val_f1}
            torch.save(model.state_dict(), best_model_path)
            print(f"  --> Saved best model (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  --> Early stopping patience: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print("Triggered early stopping!")
                break

    # ── Results ───────────────────────────────────────────────────────────────
    time_elapsed = time.time() - start_time
    print(f'\nTraining complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
    print(f"Best Val Acc: {best_metrics['acc']:.2f}%  |  Best F1: {best_metrics['f1']:.4f}")

    final_metrics = {
        "model":        args.model,
        "epochs":       total_epochs_trained,
        "lr":           args.lr,
        "batch_size":   args.batch_size,
        "label_smoothing": args.label_smoothing,
        "freeze_epochs":   args.freeze_epochs,
        "scheduler":       args.scheduler,
        "time_seconds": time_elapsed,
        "accuracy":     best_metrics['acc'],
        "precision":    best_metrics['precision'] * 100,
        "recall":       best_metrics['recall'] * 100,
        "f1_score":     best_metrics['f1'] * 100,
    }
    print("\n--- JSON_METRICS_START ---")
    print(json.dumps(final_metrics))
    print("--- JSON_METRICS_END ---")


if __name__ == '__main__':
    main()
