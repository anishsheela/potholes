import os
import glob
import json
import argparse
import uuid
import cv2
from tqdm import tqdm
from ultralytics import YOLO

def generate_label_studio_tasks(image_dir, model_path, output_json, project_root, docker_mount):
    """
    Runs YOLOv8 inference on a directory of images and exports the predictions 
    into Label Studio's "Lightweight JSON" format for pre-annotation, resolving
    broken image issues by mapping host file paths to the Docker bind mount path.
    """
    if not os.path.exists(image_dir):
        print(f"Error: Could not find image directory {image_dir}")
        return
        
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Did the Bootstrapper finish training?")
        return
        
    print(f"Loading YOLO model from {model_path}...")
    model = YOLO(model_path)
    
    images = glob.glob(os.path.join(image_dir, '**', '*.jpg'), recursive=True)
    if not images:
        print(f"No images found in {image_dir}")
        return
        
    print(f"Found {len(images)} images to auto-annotate.")
    
    label_studio_tasks = []
    
    for img_path in tqdm(images, desc="Generating Pre-Annotations"):
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        original_h, original_w = img.shape[:2]
        
        # Run inference (we use a lower confidence to capture more potentials for the human to reject)
        results = model.predict(img, conf=0.25, verbose=False)
        boxes = results[0].boxes
        
        # The user is mounting 'potholes/dataset' directly to '/label-studio/mydata' in Docker.
        # Therefore, we need the relative path starting *after* the dataset folder.
        dataset_dir = os.path.join(project_root, 'dataset')
        rel_path = os.path.relpath(img_path, dataset_dir).replace("\\", "/")
        docker_abs_path = f"{docker_mount}/{rel_path}"
        
        # Format the Local Storage URL using the absolute path inside the container
        image_url = f"/data/local-files/?d={docker_abs_path}"
        
        task = {
            "data": {
                "image": image_url
            },
            "predictions": []
        }
        
        if len(boxes) > 0:
            prediction_results = []
            
            for box in boxes:
                # YOLO format: [x_center, y_center, width, height] (normalized 0-1)
                x_c, y_c, w, h = box.xywhn[0].tolist()
                conf = float(box.conf[0].item())
                
                # Label Studio expects Top-Left X/Y in Percentages (0-100), and Width/Height in Percentages (0-100)
                ls_x = (x_c - (w / 2)) * 100
                ls_y = (y_c - (h / 2)) * 100
                ls_w = w * 100
                ls_h = h * 100
                
                # Clamp values just in case predictions bleed over the image edge
                ls_x = max(0.0, min(100.0, ls_x))
                ls_y = max(0.0, min(100.0, ls_y))
                
                prediction_results.append({
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
                    },
                    "score": conf
                })
                
            task["predictions"].append({
                "model_version": "bootstrapper-v1",
                "score": float(boxes.conf.mean().item()), 
                "result": prediction_results
            })
            
        label_studio_tasks.append(task)
        
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(label_studio_tasks, f, indent=2)
        
    print("\n--- Auto-Annotation Complete! ---")
    print(f"Exported {len(label_studio_tasks)} tasks to: {output_json}")
    print("\nLabel Studio Import Instructions (Docker):")
    print("1. In Label Studio: Settings -> Cloud Storage -> Add Source Storage.")
    print(f"2. Select 'Local Storage'. Set Absolute local path to: {docker_mount}")
    print("3. File Filter Regex: .*jpg")
    print("4. Toggle 'Treat every bucket object as a source file' ON.")
    print("5. Click 'Sync Storage'.")
    print("6. Return to your project, click 'Import', and upload this generated JSON file!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Label Studio pre-annotations from YOLO weights")
    parser.add_argument("--image-dir", "-i", type=str, default="dataset/staging", help="Folder containing images to auto-annotate")
    parser.add_argument("--weights", "-w", type=str, default="models/bootstrapper.pt", help="Path to YOLO weights to use")
    parser.add_argument("--out", "-o", type=str, default="dataset/label_studio_tasks.json", help="Output JSON map")
    parser.add_argument("--root", "-r", type=str, default=os.getcwd(), help="Absolute path to project root (for Local Storage URLs)")
    parser.add_argument("--docker-mount", "-m", type=str, default="/label-studio/mydata", help="Path where the project root is mounted inside the Docker container")
    
    args = parser.parse_args()
    
    generate_label_studio_tasks(args.image_dir, args.weights, args.out, os.path.abspath(args.root), args.docker_mount)
