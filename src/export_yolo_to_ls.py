import os
import glob
import json
import argparse
import uuid
import cv2
from tqdm import tqdm

def export_yolo_to_ls(yolo_dir, output_json, project_root, docker_mount):
    """
    Reads a directory of YOLO images and labels and exports them 
    into Label Studio's "Lightweight JSON" format for pre-annotation.
    """
    images_dir = os.path.join(yolo_dir, 'images')
    labels_dir = os.path.join(yolo_dir, 'labels')
    
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print(f"Error: Could not find 'images' or 'labels' within {yolo_dir}")
        return
        
    # Get all .txt labels
    label_files = glob.glob(os.path.join(labels_dir, '*.txt'))
    if not label_files:
        print(f"No label files found in {labels_dir}")
        return
        
    print(f"Found {len(label_files)} label files to convert.")
    
    label_studio_tasks = []
    total_potholes_converted = 0
    
    for label_path in tqdm(label_files, desc="Converting to Label Studio JSON"):
        base_name = os.path.splitext(os.path.basename(label_path))[0]
        
        # Determine original image path
        image_found = False
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            possible_path = os.path.join(images_dir, base_name + ext)
            if os.path.exists(possible_path):
                image_found = True
                img_path = possible_path
                break
                
        if not image_found:
            print(f"Warning: Corresponding image for {base_name}.txt not found.")
            continue
            
        # We need the original image dimension to calculate LS percentage coordinates
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue
            
        original_h, original_w = img.shape[:2]
        
        # Read YOLO annotations
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        if not lines:
            continue
            
        # The user's Label Studio local storage is rooted at the project folder.
        # We need the relative path starting *after* the dataset folder or project root.
        rel_path = os.path.relpath(img_path, project_root).replace("\\", "/")
        
        # Format the Local Storage URL using the relative path
        image_url = f"/data/local-files/?d={rel_path}"
        
        task = {
            "data": {
                "image": image_url
            },
            "annotations": [] 
        }
        
        annotation_results = []
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
                
            class_id = int(parts[0]) 
            x_c = float(parts[1])
            y_c = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            
            # Label Studio expects Top-Left X/Y in Percentages (0-100), and Width/Height in Percentages (0-100)
            ls_x = (x_c - (w / 2)) * 100
            ls_y = (y_c - (h / 2)) * 100
            ls_w = w * 100
            ls_h = h * 100
            
            # Clamp values just in case predictions bleed over the image edge
            ls_x = max(0.0, min(100.0, ls_x))
            ls_y = max(0.0, min(100.0, ls_y))
            
            annotation_results.append({
                "id": str(uuid.uuid4())[:10],
                "type": "rectanglelabels",
                "from_name": "label",
                "to_name": "image",
                "original_width": original_w,
                "original_height": original_h,
                "image_rotation": 0,
                "value": {
                    "rotation": 0,
                    "x": ls_x,
                    "y": ls_y,
                    "width": ls_w,
                    "height": ls_h,
                    "rectanglelabels": ["pothole"]
                }
            })
            total_potholes_converted += 1
            
        # Append as a completed annotation rather than a prediction
        if annotation_results:
            task["annotations"].append({
                "result": annotation_results,
                "was_cancelled": False,
                "ground_truth": False
            })
            
        # Even if a file is explicitly empty (which shouldn't happen here due to our filtering script),
        # Label Studio handles the empty list gracefully.
        label_studio_tasks.append(task)
        
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(label_studio_tasks, f, indent=2)
        
    print("\n--- Conversion Complete! ---")
    print(f"Exported {len(label_studio_tasks)} tasks to: {output_json}")
    print(f"Total Bounding Boxes Converted: {total_potholes_converted}")
    print("\nLabel Studio Import Instructions:")
    print("1. In Label Studio: Settings -> Cloud Storage -> Add Source Storage.")
    print(f"2. Select 'Local Storage'. Set Absolute local path to: {docker_mount}")
    print("3. File Filter Regex: .*jpg")
    print("4. Toggle 'Treat every bucket object as a source file' ON.")
    print("5. Click 'Sync Storage'.")
    print("6. Return to your project, click 'Import', and upload this JSON file!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert YOLO annotations into generic Label Studio JSON pre-annotations")
    parser.add_argument("--yolo-dir", "-i", type=str, default="dataset/review_yolo", help="Folder containing YOLO 'images' and 'labels' directories")
    parser.add_argument("--out", "-o", type=str, default="dataset/review_tasks.json", help="Output JSON map")
    parser.add_argument("--root", "-r", type=str, default=os.path.join(os.getcwd(), 'dataset'), help="Absolute path to your dataset root (this part of the path is stripped for Local Storage URLs)")
    parser.add_argument("--docker-mount", "-m", type=str, default="/label-studio/mydata", help="Path where the root is mounted inside the Docker container")
    
    args = parser.parse_args()
    
    export_yolo_to_ls(args.yolo_dir, args.out, os.path.abspath(args.root), args.docker_mount)
