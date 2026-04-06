import os
import json
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for PyTorch Image Classification from JSON labels")
    parser.add_argument('--json-file', type=str, required=True, help="Path to the classifications JSON file (e.g. from /api/export)")
    parser.add_argument('--base-dir', type=str, default="processed_data/filtered_frames", help="Base directory containing the images referenced in JSON")
    parser.add_argument('--output-dir', type=str, default="dataset/classification/training", help="Output directory structure for PyTorch")
    
    args = parser.parse_args()
    
    # Check if json file exists
    if not os.path.exists(args.json_file):
        print(f"Error: JSON file '{args.json_file}' not found.")
        return
        
    # Read the JSON classifications
    with open(args.json_file, 'r') as f:
        data = json.load(f)
        
    if not data:
        print("Error: JSON file is empty")
        return
        
    print(f"Loaded {len(data)} classifications from {args.json_file}")
    
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    
    # Create the output output directory hierarchy
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        
    # Track stats
    stats = {}
    missing_files = 0
    copied_files = 0
    
    print(f"Copying files from {base_dir} to {output_dir}/...")
    
    for item in tqdm(data):
        image_rel_path = item.get('image')
        label = item.get('label')
        
        if not image_rel_path or not label:
            continue
            
        # Clean label strings (e.g., in case there's whitespace)
        label = label.strip()
        
        # Source file path
        src_path = base_dir / image_rel_path
        
        # Destination folder
        class_dir = output_dir / label
        if not class_dir.exists():
            class_dir.mkdir(parents=True)
            stats[label] = 0
            
        # Destination file path
        dst_path = class_dir / Path(image_rel_path).name
        
        # Copy file if source exists
        if src_path.exists():
            # Only copy if dst doesn't exist to save time on reruns
            if not dst_path.exists():
                shutil.copy2(src_path, dst_path)
            copied_files += 1
            stats[label] = stats.get(label, 0) + 1
        else:
            missing_files += 1
            # print(f"Missing file: {src_path}")
            
    print("\n" + "="*40)
    print("Dataset Preparation Summary")
    print("="*40)
    print(f"Total labeled items in JSON: {len(data)}")
    print(f"Successfully copied/verified: {copied_files}")
    if missing_files > 0:
        print(f"Missing source files: {missing_files} (Check your --base-dir)")
        
    print("\nClass Distribution in Dataset:")
    for label, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {label}: {count}")
    print("="*40)
    print(f"\nYour dataset is ready at: {output_dir.absolute()}")

if __name__ == "__main__":
    main()
