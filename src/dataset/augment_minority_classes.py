"""
Offline augmentation for minority classes in the ImageFolder training dataset.

Generates augmented copies of images in under-represented classes and saves
them back into the same class folder. Safe to re-run — already-augmented
files (containing '_aug_') are never used as augmentation sources.

Usage:
    # Default targets: Poor->400, Fair->500, Invalid->400
    python src/dataset/augment_minority_classes.py

    # Custom targets
    python src/dataset/augment_minority_classes.py \
        --targets Poor:500 Fair:600 Invalid:450

    # Preview what would happen without writing any files
    python src/dataset/augment_minority_classes.py --dry-run
"""

import os
import argparse
import random
from pathlib import Path
from collections import Counter

from PIL import Image, ImageFilter
import torchvision.transforms.functional as TF
from tqdm import tqdm

# ── Default augmentation targets ──────────────────────────────────────────────
DEFAULT_TARGETS = {
    'Poor':    400,
    'Fair':    500,
    'Invalid': 400,
}

SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png'}


# ── Augmentation building blocks ──────────────────────────────────────────────

def _random_brightness(img):
    factor = random.uniform(0.70, 1.35)
    return TF.adjust_brightness(img, factor)

def _random_contrast(img):
    factor = random.uniform(0.70, 1.35)
    return TF.adjust_contrast(img, factor)

def _random_saturation(img):
    factor = random.uniform(0.75, 1.25)
    return TF.adjust_saturation(img, factor)

def _random_hue(img):
    factor = random.uniform(-0.05, 0.05)   # very mild — keeps road colours realistic
    return TF.adjust_hue(img, factor)

def _random_rotation(img):
    angle = random.uniform(-8, 8)
    return TF.rotate(img, angle, fill=0)

def _hflip(img):
    return TF.hflip(img)

def _random_blur(img):
    radius = random.uniform(0.3, 1.2)
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

def _random_crop_resize(img):
    """Zoom in by 5–15 % to simulate slight camera position variation."""
    w, h = img.size
    scale = random.uniform(0.85, 0.97)
    new_w, new_h = int(w * scale), int(h * scale)
    left  = random.randint(0, w - new_w)
    top   = random.randint(0, h - new_h)
    img = img.crop((left, top, left + new_w, top + new_h))
    return img.resize((w, h), Image.BILINEAR)

def _random_sharpness(img):
    factor = random.uniform(0.5, 2.0)
    from PIL import ImageEnhance
    return ImageEnhance.Sharpness(img).enhance(factor)


# Pool of all available transforms (each is a function PIL→PIL)
TRANSFORM_POOL = [
    _random_brightness,
    _random_contrast,
    _random_saturation,
    _random_hue,
    _random_rotation,
    _hflip,
    _random_blur,
    _random_crop_resize,
    _random_sharpness,
]


def augment_image(img: Image.Image, num_transforms: int = None) -> Image.Image:
    """
    Apply a random subset of transforms to a PIL image.
    num_transforms defaults to a random int in [2, 4] for variety.
    """
    if num_transforms is None:
        num_transforms = random.randint(2, 4)

    chosen = random.sample(TRANSFORM_POOL, k=min(num_transforms, len(TRANSFORM_POOL)))
    for fn in chosen:
        try:
            img = fn(img)
        except Exception:
            pass   # skip any transform that fails on an unusual image
    return img


# ── Core logic ─────────────────────────────────────────────────────────────────

def is_augmented(filename: str) -> bool:
    return '_aug_' in filename


def get_source_images(class_dir: Path) -> list:
    """Return only original (non-augmented) image paths in a class directory."""
    return [
        p for p in class_dir.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTS and not is_augmented(p.name)
    ]


def count_all_images(class_dir: Path) -> int:
    return sum(
        1 for p in class_dir.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTS
    )


def augment_class(class_dir: Path, target: int, dry_run: bool = False) -> int:
    """
    Generate augmented images in class_dir until count reaches target.
    Returns the number of images generated.
    """
    sources = get_source_images(class_dir)
    if not sources:
        print(f"  [!] No source images found in {class_dir}. Skipping.")
        return 0

    current_total = count_all_images(class_dir)
    needed = target - current_total

    if needed <= 0:
        print(f"  Already at or above target ({current_total} >= {target}). Skipping.")
        return 0

    print(f"  Generating {needed} augmented images  ({current_total} → {target})")

    if dry_run:
        return needed

    generated = 0
    pbar = tqdm(total=needed, unit='img', leave=False)

    # Cycle through source images so augmentations are spread evenly
    source_cycle = list(sources)
    random.shuffle(source_cycle)
    src_idx = 0

    while generated < needed:
        src_path = source_cycle[src_idx % len(source_cycle)]
        src_idx += 1

        try:
            img = Image.open(src_path).convert('RGB')
        except Exception as e:
            print(f"  [!] Could not open {src_path.name}: {e}")
            continue

        aug_img = augment_image(img)

        # Build output filename: <stem>_aug_<NNN><ext>
        stem = src_path.stem
        ext  = src_path.suffix.lower()
        # Find a unique index
        aug_idx = 1
        while True:
            out_name = f"{stem}_aug_{aug_idx:03d}{ext}"
            out_path = class_dir / out_name
            if not out_path.exists():
                break
            aug_idx += 1

        aug_img.save(out_path, quality=92)
        generated += 1
        pbar.update(1)

    pbar.close()
    return generated


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Offline augmentation for minority classes in an ImageFolder dataset'
    )
    parser.add_argument(
        '--data-dir', type=str,
        default='dataset/classification/training',
        help='Path to ImageFolder training directory'
    )
    parser.add_argument(
        '--targets', type=str, nargs='+',
        metavar='Class:N',
        help='Target image counts per class, e.g. --targets Poor:500 Fair:600. '
             'Defaults: Poor:400 Fair:500 Invalid:400'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Print what would be done without writing any files'
    )
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: {data_dir} does not exist.")
        return

    # Parse targets
    targets = dict(DEFAULT_TARGETS)
    if args.targets:
        for item in args.targets:
            if ':' not in item:
                print(f"Warning: ignoring malformed target '{item}' (expected Class:N)")
                continue
            cls, n = item.split(':', 1)
            try:
                targets[cls.strip()] = int(n.strip())
            except ValueError:
                print(f"Warning: ignoring non-integer target '{item}'")

    # Discover classes present in the directory
    available_classes = [d.name for d in data_dir.iterdir() if d.is_dir()]

    print("\n─── Before Augmentation ───────────────────────────────────")
    before = {}
    for cls in sorted(available_classes):
        count = count_all_images(data_dir / cls)
        before[cls] = count
        marker = ' ← will augment' if cls in targets else ''
        print(f"  {cls:<12} {count:>5} images{marker}")

    if args.dry_run:
        print("\n[DRY RUN] No files will be written.\n")

    print("\n─── Augmenting ─────────────────────────────────────────────")
    total_generated = 0
    for cls, target in targets.items():
        class_dir = data_dir / cls
        if not class_dir.exists():
            print(f"\n  [{cls}] Directory not found — skipping.")
            continue
        print(f"\n  [{cls}]  target = {target}")
        generated = augment_class(class_dir, target, dry_run=args.dry_run)
        total_generated += generated

    print("\n─── After Augmentation ─────────────────────────────────────")
    for cls in sorted(available_classes):
        count_before = before[cls]
        count_after  = count_all_images(data_dir / cls) if not args.dry_run else (
            count_before + max(0, targets.get(cls, count_before) - count_before)
        )
        delta = count_after - count_before
        delta_str = f'(+{delta})' if delta > 0 else ''
        print(f"  {cls:<12} {count_after:>5} images  {delta_str}")

    suffix = ' (dry run)' if args.dry_run else ''
    print(f"\nDone{suffix}. Total augmented images generated: {total_generated}\n")


if __name__ == '__main__':
    main()
