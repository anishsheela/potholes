#!/usr/bin/env python3
import os
import argparse
import sqlite3
import shutil
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="Export Consensus Images to Training Folder")
    parser.add_argument('--db', type=str, default='road_classifier/classifications.db', help='Path to sqlite database')
    parser.add_argument('--data-dir', type=str, default='processed_data/frames', help='Root directory where images live')
    parser.add_argument('--out-dir', type=str, default='dataset/classification/training', help='Output directory for ImageFolder')
    parser.add_argument('--clean', action='store_true', help='Delete existing output directory before exporting')
    return parser.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(args.db):
        print(f"Error: Database {args.db} not found!")
        return

    # Optional: Wipe the old dataset so we don't bleed old inconsistent annotations
    if args.clean and os.path.exists(args.out_dir):
        print(f"Cleaning existing directory {args.out_dir}...")
        shutil.rmtree(args.out_dir)

    conn = sqlite3.connect(args.db)
    c = conn.cursor()
    c.execute('SELECT image_name, username, label FROM classifications')
    rows = c.fetchall()
    conn.close()

    # Dictionary mapping image_name -> list of labels
    img_labels = defaultdict(list)
    for img, user, label in rows:
        img_labels[img].append(label)

    # Find images with Consensus (at least 2 identical labels)
    consensus_data = []
    
    for img, labels in img_labels.items():
        unique_labels = set(labels)
        for label in unique_labels:
            if labels.count(label) >= 2:
                consensus_data.append((img, label))
                break # Move to next image once consensus is found

    print(f"Found {len(consensus_data)} images that reached consensus.")
    
    if len(consensus_data) == 0:
        print("No images have reached a 2-vote consensus yet! Get your friends to annotate more.")
        return

    # Perform the export
    os.makedirs(args.out_dir, exist_ok=True)
    exported_counts = defaultdict(int)

    for img_rel_path, final_label in consensus_data:
        src_path = os.path.join(args.data_dir, img_rel_path)
        
        # If the web app was pointing at active_learning_images or unfiltered_images, 
        # the file name might look different, so we check if it exists there, 
        # or we check if it was already flattened.
        if not os.path.exists(src_path):
            # Try exploring alternative root directories if data-dir was switched mid-stream
            alt_src1 = os.path.join("road_classifier/unfiltered_images", img_rel_path)
            alt_src2 = os.path.join("road_classifier/active_learning_images", img_rel_path)
            
            if os.path.exists(alt_src1):
                src_path = alt_src1
            elif os.path.exists(alt_src2):
                src_path = alt_src2
            else:
                print(f"Warning: Could not locate source image for {img_rel_path}. Skipping.")
                continue

        # Flatten output filename to avoid subfolder clashing
        flat_name = img_rel_path.replace(os.sep, '_')
        
        # Class directory inside out_dir (e.g., dataset/classification/training/Excellent)
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
    print("\nYou can now run:")
    print("  python src/train_classifier.py --data-dir dataset/classification/training")

if __name__ == '__main__':
    main()
