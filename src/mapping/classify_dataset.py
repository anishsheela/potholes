"""
Step 1 of the road quality mapping pipeline.

Runs the trained classifier over all filtered dashcam frames and joins each
prediction with its GPS coordinates from the route CSVs.

Output CSV columns:
    frame_path, video, seconds, latitude, longitude,
    predicted_class, confidence, prob_excellent, prob_fair,
    prob_good, prob_invalid, prob_poor

Usage:
    python src/mapping/classify_dataset.py \
        --weights models/swin_small_patch4_window7_224_best.pth \
        --output processed_data/mapping/predictions.csv

    # Drop Invalid predictions and low-confidence frames:
    python src/mapping/classify_dataset.py \
        --weights models/swin_small_patch4_window7_224_best.pth \
        --conf-threshold 0.70 \
        --drop-invalid \
        --output processed_data/mapping/predictions.csv
"""

import os
import re
import csv
import argparse
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
from tqdm import tqdm

# Class order matches PyTorch ImageFolder alphabetical sort when trained
# with --include-invalid (5 classes) or without (4 classes).
CLASSES_5 = ['Excellent', 'Fair', 'Good', 'Invalid', 'Poor']
CLASSES_4 = ['Excellent', 'Fair', 'Good', 'Poor']


# ── GPS helpers ───────────────────────────────────────────────────────────────

def parse_coord(value: str) -> float:
    """Convert 'N8.5010' / 'E76.9478' / 'S...' / 'W...' to signed float."""
    value = value.strip()
    if not value:
        return float('nan')
    sign = -1 if value[0] in ('S', 'W') else 1
    return sign * float(value[1:])


def load_gps_index(gps_dir: str) -> dict:
    """
    Load all filtered_*.csv files and build a lookup dict:
        (video_name, seconds_float) -> (latitude, longitude)
    """
    index = {}
    pattern = os.path.join(gps_dir, 'filtered_*.csv')
    csv_files = glob.glob(pattern)

    if not csv_files:
        # Fall back to any *.csv in the directory
        csv_files = glob.glob(os.path.join(gps_dir, '*.csv'))

    for csv_path in csv_files:
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    video = row['Video'].strip()
                    seconds = float(row['Video_Seconds'])
                    lat = parse_coord(row['Latitude'])
                    lon = parse_coord(row['Longitude'])
                    index[(video, seconds)] = (lat, lon)
                except (KeyError, ValueError):
                    continue

    print(f"GPS index loaded: {len(index):,} entries from {len(csv_files)} CSV file(s)")
    return index


# ── Frame discovery ───────────────────────────────────────────────────────────

_FRAME_RE = re.compile(r'frame_(\d+(?:\.\d+)?)s\.jpg$', re.IGNORECASE)


def parse_frame_path(path: str):
    """
    Extract (video_name, seconds) from a frame path like:
        .../filtered_frames/anish/NO20260217-162918-006762F/frame_0044.0s.jpg
    Returns (video_name, seconds_float) or (None, None) on parse failure.
    """
    m = _FRAME_RE.search(os.path.basename(path))
    if not m:
        return None, None
    seconds = float(m.group(1))
    video = os.path.basename(os.path.dirname(path))
    return video, seconds


def discover_frames(frames_dir: str) -> list:
    """Walk frames_dir and return all .jpg frame paths."""
    paths = []
    for root, _, files in os.walk(frames_dir):
        for f in files:
            if f.lower().endswith('.jpg') and _FRAME_RE.search(f):
                paths.append(os.path.join(root, f))
    return sorted(paths)


# ── Dataset ───────────────────────────────────────────────────────────────────

class FrameDataset(Dataset):
    def __init__(self, paths: list, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert('RGB')
            return self.transform(img), path
        except Exception:
            return torch.zeros(3, 224, 224), path


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Classify all filtered frames and attach GPS coordinates')
    parser.add_argument('--frames-dir', type=str, default='processed_data/filtered_frames',
                        help='Root directory containing filtered frames (organised by user/video)')
    parser.add_argument('--gps-dir', type=str, default='processed_data/route_data',
                        help='Directory containing filtered_*.csv GPS route files')
    parser.add_argument('--model', type=str, default='swin_small_patch4_window7_224',
                        help='timm model architecture used during training')
    parser.add_argument('--weights', type=str,
                        default='models/swin_small_patch4_window7_224_best.pth',
                        help='Path to trained model weights (.pth)')
    parser.add_argument('--output', type=str, default='processed_data/mapping/predictions.csv',
                        help='Output CSV path')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--conf-threshold', type=float, default=0.0,
                        help='Drop predictions below this confidence (0–1). Default: keep all.')
    parser.add_argument('--drop-invalid', action='store_true',
                        help='Exclude frames predicted as Invalid from the output')
    parser.add_argument('--no-gps-drop', action='store_true',
                        help='Keep rows even when no GPS match is found (lat/lon will be empty)')
    parser.add_argument('--num-classes', type=int, default=5, choices=[4, 5],
                        help='Number of output classes the model was trained with (4 without Invalid, 5 with)')
    return parser.parse_args()


def main():
    args = parse_args()

    classes = CLASSES_5 if args.num_classes == 5 else CLASSES_4

    # ── Validate inputs ───────────────────────────────────────────────────────
    if not os.path.exists(args.weights):
        print(f"Error: weights not found at {args.weights}")
        return
    if not os.path.exists(args.frames_dir):
        print(f"Error: frames directory not found at {args.frames_dir}")
        return

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps'  if torch.backends.mps.is_available() else
        'cpu'
    )
    print(f"Device: {device}")

    # ── GPS index ─────────────────────────────────────────────────────────────
    gps_index = load_gps_index(args.gps_dir)

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"Loading {args.model} ({args.num_classes} classes)...")
    model = timm.create_model(args.model, pretrained=False, num_classes=args.num_classes)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    print("Model loaded.")

    # ── Transforms (same as val transforms in train_classifier.py) ────────────
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # ── Discover frames ───────────────────────────────────────────────────────
    print(f"Scanning {args.frames_dir}...")
    all_paths = discover_frames(args.frames_dir)
    print(f"Found {len(all_paths):,} frames.")

    if not all_paths:
        print("No frames found. Exiting.")
        return

    dataset = FrameDataset(all_paths, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers,
                        pin_memory=(device.type == 'cuda'))

    # ── Inference ─────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    fieldnames = [
        'frame_path', 'video', 'seconds', 'latitude', 'longitude',
        'predicted_class', 'confidence',
    ] + [f'prob_{c.lower()}' for c in classes]

    total_written = 0
    skipped_conf = 0
    skipped_invalid = 0
    skipped_gps = 0
    class_counts = {c: 0 for c in classes}

    with open(args.output, 'w', newline='') as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()

        with torch.no_grad():
            for imgs, paths in tqdm(loader, desc='Classifying'):
                imgs = imgs.to(device)
                logits = model(imgs)
                probs = F.softmax(logits, dim=1).cpu()

                for i, path in enumerate(paths):
                    prob_vec = probs[i]
                    conf, pred_idx = prob_vec.max(0)
                    conf = conf.item()
                    predicted = classes[pred_idx.item()]

                    # Confidence filter
                    if conf < args.conf_threshold:
                        skipped_conf += 1
                        continue

                    # Invalid filter
                    if args.drop_invalid and predicted == 'Invalid':
                        skipped_invalid += 1
                        continue

                    # GPS lookup
                    video, seconds = parse_frame_path(path)
                    lat, lon = None, None
                    if video is not None:
                        coords = gps_index.get((video, seconds))
                        if coords:
                            lat, lon = coords

                    if lat is None and not args.no_gps_drop:
                        skipped_gps += 1
                        continue

                    row = {
                        'frame_path':      path,
                        'video':           video or '',
                        'seconds':         seconds if seconds is not None else '',
                        'latitude':        f'{lat:.6f}' if lat is not None else '',
                        'longitude':       f'{lon:.6f}' if lon is not None else '',
                        'predicted_class': predicted,
                        'confidence':      f'{conf:.4f}',
                    }
                    for j, cls in enumerate(classes):
                        row[f'prob_{cls.lower()}'] = f'{prob_vec[j].item():.4f}'

                    writer.writerow(row)
                    class_counts[predicted] += 1
                    total_written += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f'\n{"="*55}')
    print(f'Classification complete')
    print(f'{"="*55}')
    print(f'  Total frames processed : {len(all_paths):,}')
    print(f'  Written to CSV         : {total_written:,}')
    if skipped_conf:
        print(f'  Skipped (low conf<{args.conf_threshold:.2f}): {skipped_conf:,}')
    if skipped_invalid:
        print(f'  Skipped (Invalid class): {skipped_invalid:,}')
    if skipped_gps:
        print(f'  Skipped (no GPS match) : {skipped_gps:,}')
    print(f'\nPrediction distribution:')
    for cls in classes:
        pct = 100 * class_counts[cls] / max(total_written, 1)
        print(f'  {cls:<12} {class_counts[cls]:>6,}  ({pct:.1f}%)')
    print(f'\nOutput: {args.output}')


if __name__ == '__main__':
    main()
