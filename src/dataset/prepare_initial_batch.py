#!/usr/bin/env python3
import os
import argparse
import random
import shutil
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Initial 500 Image Batch (Oversampling Rajeshe)")
    parser.add_argument('--data-dir', type=str, default='processed_data/frames', help='Root dir containing dashcam folders')
    parser.add_argument('--out-dir', type=str, default='road_classifier/unfiltered_images', help='Output dir for the UI to consume')
    parser.add_argument('--total', type=int, default=500, help='Total images to sample')
    parser.add_argument('--rajeshe-ratio', type=float, default=0.5, help='Fraction of the total to pull from the rajeshe events folder')
    return parser.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(args.data_dir):
        print(f"Error: Directory {args.data_dir} not found.")
        return

    rajeshe_dir = os.path.join(args.data_dir, 'rajeshe')
    other_folders = [os.path.join(args.data_dir, d) for d in os.listdir(args.data_dir) 
                     if os.path.isdir(os.path.join(args.data_dir, d)) and d != 'rajeshe']

    valid_exts = {'.png', '.jpg', '.jpeg', '.bmp'}

    def get_all_images(folder):
        images = []
        if os.path.exists(folder):
            for root, _, files in os.walk(folder):
                for f in files:
                    if os.path.splitext(f)[1].lower() in valid_exts:
                        images.append(os.path.join(root, f))
        return images

    print("Scanning folders...")
    rajeshe_imgs = get_all_images(rajeshe_dir)
    other_imgs = []
    for d in other_folders:
        other_imgs.extend(get_all_images(d))

    print(f"Found {len(rajeshe_imgs)} in 'rajeshe' (events)")
    print(f"Found {len(other_imgs)} in all other folders")

    num_rajeshe = int(args.total * args.rajeshe_ratio)
    num_other = args.total - num_rajeshe

    # Guard against not having enough images
    num_rajeshe = min(num_rajeshe, len(rajeshe_imgs))
    num_other = min(num_other, len(other_imgs))

    sampled_rajeshe = random.sample(rajeshe_imgs, num_rajeshe)
    sampled_other = random.sample(other_imgs, num_other)

    final_sample = sampled_rajeshe + sampled_other
    random.shuffle(final_sample)

    print(f"\nSampling {len(sampled_rajeshe)} from rajeshe and {len(sampled_other)} from others -> Total: {len(final_sample)}")

    os.makedirs(args.out_dir, exist_ok=True)
    
    # Optional: Clear out the existing directory or just add? 
    # Let's just create it and copy.
    copied = 0
    for path in final_sample:
        # Flatten names to avoid subdirectory clashing
        rel_path = os.path.relpath(path, args.data_dir)
        flat_name = rel_path.replace(os.path.sep, '_')
        
        dest = os.path.join(args.out_dir, flat_name)
        shutil.copy2(path, dest)
        copied += 1
        
    print(f"Successfully copied {copied} images to {args.out_dir}!")
    print("\nYou can now start the web UI:")
    print(f"  cd road_classifier && python app.py --data-dir unfiltered_images")

if __name__ == '__main__':
    main()
