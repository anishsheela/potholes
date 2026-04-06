import os
import shutil
import argparse
from tqdm import tqdm

def filter_non_empty_yolo(input_dir, output_dir):
    """
    Scans a YOLO dataset directory, finds all non-empty label files,
    and copies them along with their corresponding image files to a new directory.
    """
    input_images_dir = os.path.join(input_dir, "images")
    input_labels_dir = os.path.join(input_dir, "labels")
    
    if not os.path.exists(input_labels_dir):
        print(f"Error: Could not find labels directory at {input_labels_dir}")
        return
        
    output_images_dir = os.path.join(output_dir, "images")
    output_labels_dir = os.path.join(output_dir, "labels")
    
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    
    label_files = [f for f in os.listdir(input_labels_dir) if f.endswith(".txt")]
    
    if not label_files:
        print(f"No label files found in {input_labels_dir}")
        return
        
    print(f"Found {len(label_files)} label files to process.")
    
    copied_count = 0
    
    for label_file in tqdm(label_files, desc="Filtering non-empty labels"):
        label_path = os.path.join(input_labels_dir, label_file)
        
        # Check if file has size > 0 and contains at least one non-whitespace character
        is_empty = True
        try:
            if os.path.getsize(label_path) > 0:
                with open(label_path, 'r') as f:
                    content = f.read().strip()
                    if content:
                        is_empty = False
        except Exception as e:
            print(f"Error reading file {label_path}: {e}")
            continue
            
        if not is_empty:
            # Copy label
            output_label_path = os.path.join(output_labels_dir, label_file)
            shutil.copy2(label_path, output_label_path)
            
            # Find and copy corresponding image
            # The image might have different extensions (.jpg, .png, .jpeg)
            base_name = os.path.splitext(label_file)[0]
            image_found = False
            
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                image_file = base_name + ext
                image_path = os.path.join(input_images_dir, image_file)
                
                if os.path.exists(image_path):
                    output_image_path = os.path.join(output_images_dir, image_file)
                    shutil.copy2(image_path, output_image_path)
                    image_found = True
                    break
            
            if not image_found:
                print(f"Warning: Corresponding image for {label_file} not found.")
            else:
                copied_count += 1
                
    # Create classes.txt in the output root folder
    classes_path = os.path.join(output_dir, "classes.txt")
    with open(classes_path, 'w') as f:
        f.write("pothole\n")
        
    print(f"\n--- Filtering Complete! ---")
    print(f"Copied {copied_count} non-empty labels and images to {output_dir}")
    print(f"A classes.txt file has been created at {classes_path}")
    print("\nNext Steps for Label Studio:")
    print(f"1. Navigate to {output_dir}")
    print("2. Select 'images' and 'labels' folders, and 'classes.txt', then compress them into a .zip file.")
    print("3. In Label Studio, create a new project and configure the UI for Object Detection.")
    print("4. Click 'Import' and upload the .zip file.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter non-empty YOLO labels for easy review")
    parser.add_argument("--input", "-i", type=str, default="dataset/training", help="Input YOLO dataset directory containing 'images' and 'labels' folders")
    parser.add_argument("--output", "-o", type=str, default="dataset/review_yolo", help="Output directory for filtered data")
    
    args = parser.parse_args()
    
    filter_non_empty_yolo(args.input, args.output)
