#!/usr/bin/env python3
import os
import argparse
import sqlite3
import shutil
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="Export Consensus Images to Training Folder")
    parser.add_argument('--db', type=str, default='road_classifier/classifications.db', help='Path to sqlite database')
    parser.add_argument('--out-dir', type=str, default='dataset/classification/training', help='Output directory for ImageFolder')
    parser.add_argument('--clean', action='store_true', help='Delete existing output directory before exporting')
    return parser.parse_args()

def build_file_map():
    # Directories to scan for images
    search_dirs = [
        'processed_data/frames',
        'unfiltered_images',
        'road_classifier/unfiltered_images',
        'road_classifier/active_learning_images'
    ]
    file_map = {}
    for d in search_dirs:
        if os.path.exists(d):
            for root, _, files in os.walk(d):
                for f in files:
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        file_map[f] = os.path.join(root, f)
    return file_map

def main():
    args = parse_args()

    if not os.path.exists(args.db):
        print(f"Error: Database {args.db} not found!")
        return

    if args.clean and os.path.exists(args.out_dir):
        print(f"Cleaning existing directory {args.out_dir}...")
        shutil.rmtree(args.out_dir)

    conn = sqlite3.connect(args.db)
    c = conn.cursor()
    c.execute('SELECT image_name, username, label FROM classifications')
    rows = c.fetchall()
    conn.close()

    img_labels = defaultdict(list)
    for img, user, label in rows:
        img_labels[img].append(label)

    consensus_data = []
    for img, labels in img_labels.items():
        unique_labels = set(labels)
        for label in unique_labels:
            if labels.count(label) >= 2:
                consensus_data.append((img, label))
                break

    print(f"Found {len(consensus_data)} images that reached consensus.")
    if len(consensus_data) == 0:
        print("No images have reached a 2-vote consensus yet!")
        return

    # To avoid "file not found" issues if paths changed, build a smart index
    print("Building smart index of all available image files...")
    file_map = build_file_map()

    os.makedirs(args.out_dir, exist_ok=True)
    exported_counts = defaultdict(int)
    missing_count = 0

    for img_rel_path, final_label in consensus_data:
        basename = os.path.basename(img_rel_path)
        
        # 1. Try smart lookup
        src_path = file_map.get(basename)
        
        # 2. Try exact path from root
        if not src_path and os.path.exists(img_rel_path):
            src_path = img_rel_path
            
        if not src_path:
            print(f"Warning: Could not locate source image for {img_rel_path} anywhere. Skipping.")
            missing_count += 1
            continue

        flat_name = img_rel_path.replace(os.sep, '_').replace('..', '')
        class_dir = os.path.join(args.out_dir, final_label)
        os.makedirs(class_dir, exist_ok=True)
        
        dst_path = os.path.join(class_dir, flat_name)
        shutil.copy2(src_path, dst_path)
        exported_counts[final_label] += 1

    print("\n--- Export Summary ---")
    for cls_name, count in exported_counts.items():
        print(f"  {cls_name}: {count} images")
    
    total = sum(exported_counts.values())
    print(f"\nSuccessfully exported {total} images to {args.out_dir}")
    if missing_count > 0:
        print(f"Warning: {missing_count} images were completely missing from disk.")
    print("\nYou can now run:")
    print("  python src/train_classifier.py --data-dir dataset/classification/training")

if __name__ == '__main__':
    main()
