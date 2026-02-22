import os
import glob
import argparse
from ultralytics import YOLOWorld
from tqdm import tqdm

def auto_label(images_dir, labels_dir, confidence=0.05):
    """
    Uses YOLO-World to perform zero-shot detection of potholes
    and saves the bounding boxes in YOLO (.txt) format.
    """
    # Make sure output directory exists
    os.makedirs(labels_dir, exist_ok=True)
    
    # Find all jpegs in the input directory
    image_files = glob.glob(os.path.join(images_dir, "*.jpg"))
    if not image_files:
        print(f"Error: No images found in {images_dir}")
        return
        
    print(f"Found {len(image_files)} images to auto-label.")
    
    # Load a YOLO-World model
    # yolov8s-worldv2.pt is the small version, good balance of speed and accuracy
    print("Loading YOLO-World model...")
    model = YOLOWorld('yolov8s-worldv2.pt') 
    
    # Define our custom classes. YOLO-World will search for these open-vocabulary terms.
    classes = ["pothole"]
    model.set_classes(classes)
    
    print(f"Detecting '{classes[0]}' with confidence threshold {confidence}...")
    
    labels_generated = 0
    total_boxes = 0

    # Run inference on all images
    # We disable verbose mode to keep the console clean and use tqdm instead
    for img_path in tqdm(image_files, desc="Processing Images"):
        # Run YOLO-World
        # YOLO-World tends to be conservative, so we use a lower confidence 
        # threshold for the initial draft to ensure we catch most potholes.
        # The user will manually delete false positives later in MakeSense.ai.
        results = model.predict(img_path, conf=confidence, verbose=False)
        
        # We only process the first result since we pass one image at a time
        result = results[0]
        
        # If potholes were found, create the .txt file
        if len(result.boxes) > 0:
            filename = os.path.basename(img_path)
            base_name = os.path.splitext(filename)[0]
            txt_path = os.path.join(labels_dir, f"{base_name}.txt")
            
            with open(txt_path, 'w') as f:
                for box in result.boxes:
                    # YOLO format: class_id x_center y_center width height (normalized)
                    class_id = int(box.cls[0].item())
                    x, y, w, h = box.xywhn[0].tolist()
                    f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                    total_boxes += 1
            
            labels_generated += 1

    print(f"\n--- Auto-Labeling Summary ---")
    print(f"Images Processed:  {len(image_files)}")
    print(f"Files Generated:   {labels_generated} (.txt files)")
    print(f"Potholes Drafted:  {total_boxes}")
    print(f"Saved YOLO labels to {labels_dir}")
    print("\nNext step: Upload both the images folder and labels folder to MakeSense.ai for review!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-shot Auto-Labeling using YOLO-World.")
    parser.add_argument("--images-dir", type=str, default="dataset/images", help="Directory containing images to label")
    parser.add_argument("--labels-dir", type=str, default="dataset/labels", help="Output directory for YOLO format .txt files")
    parser.add_argument("--conf", type=float, default=0.05, help="Confidence threshold for predictions. Keep low for drafting.")
    
    args = parser.parse_args()
    
    auto_label(args.images_dir, args.labels_dir, args.conf)
