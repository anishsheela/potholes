import os
import glob
import shutil
import zipfile
import argparse

def sync_label_studio_export(zip_path, dataset_dir="dataset"):
    """
    Takes a YOLO format ZIP export from Label Studio.
    1. Extracts the labels.
    2. Finds the corresponding original images in dataset/unannotated/.
    3. Moves both the label and the image together into a new dataset/annotated/batch_X/ folder.
    """
    if not os.path.exists(zip_path):
        print(f"Error: Could not find export ZIP at {zip_path}")
        return
        
    print(f"Syncing annotations from {zip_path}...")
    
    staging_dir = os.path.join(dataset_dir, "staging")
    training_dir = os.path.join(dataset_dir, "training")
    
    # Ensure target directories exist
    target_images = os.path.join(training_dir, "images")
    target_labels = os.path.join(training_dir, "labels")
    os.makedirs(target_images, exist_ok=True)
    os.makedirs(target_labels, exist_ok=True)
    
    temp_extract_dir = os.path.join(dataset_dir, "temp_export")
    os.makedirs(temp_extract_dir, exist_ok=True)
    
    try:
        # 1. Unzip the Label Studio export
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_extract_dir)
            
        # Label Studio YOLO export structure:
        # temp_export/
        #   classes.txt
        #   labels/
        #     image1.txt
        #     image2.txt
        
        extracted_labels_dir = os.path.join(temp_extract_dir, "labels")
        if not os.path.exists(extracted_labels_dir):
            print("Error: The ZIP file does not appear to be in 'YOLO' format. Missing 'labels/' folder.")
            return
            
        label_files = glob.glob(os.path.join(extracted_labels_dir, "*.txt"))
        
        if not label_files:
            print("No annotations found in the ZIP.")
            return
            
        print(f"Found {len(label_files)} annotation files.")
        
        synced_count = 0
        missing_images = 0
        
        # 2. Iterate through labels, find matching image, and move both
        for label_path in label_files:
            base_filename = os.path.splitext(os.path.basename(label_path))[0]
            
            # Label Studio sometimes appends an ID to the filename (e.g., '12345-image.jpg').
            # We try to strict match first, then fallback to partial matching if they use LS default storage.
            
            # Search for the original image in the staging directory
            image_search_pattern = os.path.join(staging_dir, "**", f"{base_filename}.jpg")
            found_images = glob.glob(image_search_pattern, recursive=True)
            
            # If not found strict, try to strip the Label Studio hash prefix if it exists
            if not found_images and "-" in base_filename:
                stripped_name = base_filename.split("-", 1)[1]
                fallback_pattern = os.path.join(staging_dir, "**", f"{stripped_name}.jpg")
                found_images = glob.glob(fallback_pattern, recursive=True)
                
            if found_images:
                original_img_path = found_images[0]
                
                # Move the Image to training/
                dest_img_path = os.path.join(target_images, os.path.basename(original_img_path))
                shutil.move(original_img_path, dest_img_path)
                
                # Copy the Label to training/ (using the exact original image name to be safe)
                final_label_name = os.path.splitext(os.path.basename(original_img_path))[0] + ".txt"
                dest_label_path = os.path.join(target_labels, final_label_name)
                shutil.copy2(label_path, dest_label_path)
                
                synced_count += 1
            else:
                missing_images += 1
                
        print(f"\n--- Sync Complete! ---")
        print(f"Successfully moved {synced_count} images from 'staging/' to 'training/'.")
        if missing_images > 0:
            print(f"Warning: Could not find original images for {missing_images} labels in 'staging/'.")
            
        print("\nYou are now ready to retrain your custom model!")
        print("Run: python src/train_yolo.py --data-dir dataset")
        
    finally:
        # Cleanup
        if os.path.exists(temp_extract_dir):
            shutil.rmtree(temp_extract_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync Label Studio YOLO Zip export into the dataset structure.")
    parser.add_argument("--zip", "-z", type=str, required=True, help="Path to the downloaded Label Studio ZIP export")
    parser.add_argument("--dataset", "-d", type=str, default="dataset", help="Base dataset directory")
    
    args = parser.parse_args()
    
    sync_label_studio_export(args.zip, args.dataset)
