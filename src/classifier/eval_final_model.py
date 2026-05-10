import torch
import timm
import os
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate final merged-class Swin-Small model on val set')
    parser.add_argument('--data-dir', type=str, default='dataset/classification/training')
    parser.add_argument('--weights', type=str, default='models/swin_small_patch4_window7_224_best.pth')
    parser.add_argument('--val-split', type=float, default=0.3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Data:   {args.data_dir}")
    print(f"Weights:{args.weights}")

    MERGE = {'Excellent': 'Good', 'Good': 'Good', 'Fair': 'Bad', 'Poor': 'Bad'}
    classes_out = ['Bad', 'Good']

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_ds = datasets.ImageFolder(args.data_dir)
    print(f"\nOriginal classes: {full_ds.classes}")

    label_map = {}
    for old_cls, old_idx in full_ds.class_to_idx.items():
        new_cls = MERGE.get(old_cls, old_cls)
        if new_cls in classes_out:
            label_map[old_idx] = classes_out.index(new_cls)

    samples = [(p, label_map[l]) for p, l in full_ds.samples if l in label_map]
    print(f"Samples after merge (Invalid excluded): {len(samples)}")

    groups = [
        os.path.basename(p).split('_frame_')[0] if '_frame_' in p else os.path.basename(p)
        for p, _ in samples
    ]
    labels = [l for _, l in samples]

    gss = GroupShuffleSplit(n_splits=1, test_size=args.val_split, random_state=args.seed)
    _, val_idx = next(gss.split(range(len(samples)), labels, groups))
    print(f"Val set size: {len(val_idx)}")

    class SimpleDS(torch.utils.data.Dataset):
        def __init__(self, samples, transform):
            self.samples = samples
            self.transform = transform
            self.loader = datasets.folder.default_loader

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, i):
            p, l = self.samples[i]
            return self.transform(self.loader(p)), l

    val_ds = SimpleDS([samples[i] for i in val_idx], val_tf)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = timm.create_model(
        'swin_small_patch4_window7_224', pretrained=False, num_classes=2
    ).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()
    print("\nModel loaded. Running inference...")

    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in val_loader:
            preds = model(x.to(device)).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y.numpy())

    print("\n" + "=" * 50)
    print("Classification Report")
    print("=" * 50)
    print(classification_report(all_targets, all_preds, target_names=classes_out))

    cm = confusion_matrix(all_targets, all_preds)
    print("Confusion Matrix (rows=True, cols=Predicted)")
    print(f"{'':10} {'Bad':>6} {'Good':>6}")
    for cls, row in zip(classes_out, cm):
        print(f"{cls:<10} {row[0]:>6} {row[1]:>6}")


if __name__ == '__main__':
    main()
