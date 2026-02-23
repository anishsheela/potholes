import os
import glob
import cv2
import argparse
import shutil
import easyocr
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

def anonymize_dataset(dataset_dir, out_dir, conf_threshold=0.25):
    """
    Clones the raw dataset to a public directory, runs YOLOv8n to detect people 
    and vehicles, and applies a Gaussian blur to anonymize them.
    - People/Bicycles (class 0, 1): Full bounding box blur
    - Vehicles (2, 3, 5, 7): OCR scans for text (numberplates) and blurs only the text box
    """
    if not os.path.exists(dataset_dir):
        print(f"Error: Source directory '{dataset_dir}' does not exist.")
        return
        
    if not os.path.exists(out_dir):
        print(f"Cloning raw dataset from '{dataset_dir}' to '{out_dir}'...")
        shutil.copytree(dataset_dir, out_dir)
    else:
        print(f"Directory '{out_dir}' already exists. Anonymizing files inside it.")
        
    print(f"Loading YOLOv8n object detector...")
    model = YOLO('yolov8n.pt')
    
    print(f"Loading EasyOCR completely for targeted License Plate blurring (this may take a moment)...")
    import torch
    use_gpu = torch.backends.mps.is_available() or torch.cuda.is_available()
    reader = easyocr.Reader(['en'], gpu=use_gpu)
    
    # 0 = person, 1 = bicycle (Blur these entirely)
    person_classes = [0, 1]
    # 2 = car, 3 = motorcycle, 5 = bus, 7 = truck (Run OCR to find plates)
    vehicle_classes = [2, 3, 5, 7]
    all_target_classes = person_classes + vehicle_classes
    
    images = glob.glob(os.path.join(out_dir, '**', '*.jpg'), recursive=True)
    if not images:
        print(f"No images found in {out_dir}")
        return
        
    print(f"Found {len(images)} images to process.")
    
    for img_path in tqdm(images, desc="Anonymizing"):
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        h, w = img.shape[:2]
        
        # Run YOLO inference
        results = model.predict(img, conf=conf_threshold, classes=all_target_classes, verbose=False)
        
        boxes = results[0].boxes
        if len(boxes) == 0:
            continue
            
        for box in boxes:
            cls = int(box.cls[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x1 >= x2 or y1 >= y2:
                continue
                
            if cls in person_classes:
                # FULL BLUR: Pedestrians and Bicyclists are fully blurred to protect identity
                roi = img[y1:y2, x1:x2]
                blurred = cv2.GaussianBlur(roi, (51, 51), 0)
                img[y1:y2, x1:x2] = blurred
                
            elif cls in vehicle_classes:
                # TARGETED BLUR: Only blur Numberplates on Cars/Trucks/Motorcycles
                vehicle_roi = img[y1:y2, x1:x2]
                
                # Run OCR specifically against the cropped vehicle
                plate_hits = reader.readtext(vehicle_roi, min_size=10, text_threshold=0.3)
                
                for hit in plate_hits:
                    bbox, text, prob = hit
                    # EasyOCR bbox format: [[top_left, top_right, bottom_right, bottom_left]]
                    # We just need the min/max X and Y to define a simple rectangle
                    coords = np.array(bbox, dtype=np.int32)
                    px1 = max(0, np.min(coords[:, 0]))
                    py1 = max(0, np.min(coords[:, 1]))
                    px2 = min(x2 - x1, np.max(coords[:, 0]))
                    py2 = min(y2 - y1, np.max(coords[:, 1]))
                    
                    if px1 >= px2 or py1 >= py2:
                        continue
                        
                    # Extract just the specific text bounding box out of the vehicle ROI
                    plate_roi = vehicle_roi[py1:py2, px1:px2]
                    
                    # Apply a heavy blur to just that tiny license plate rectangle
                    try:
                        blurred_plate = cv2.GaussianBlur(plate_roi, (31, 31), 0)
                        
                        # Map it back to the absolute image coordinates to paste it in
                        abs_x1, abs_y1 = x1 + px1, y1 + py1
                        abs_x2, abs_y2 = x1 + px2, y1 + py2
                        
                        img[abs_y1:abs_y2, abs_x1:abs_x2] = blurred_plate
                    except Exception as e:
                        # Skip if the bounding box was too small/weird for GaussianBlur
                        pass
                
        # Save output
        cv2.imwrite(img_path, img)

    print("\n--- Anonymization Complete ---")
    print(f"All images in '{out_dir}' have had their pedestrians and vehicle license plates blurred.")
    print(f"Your raw, pristine training data in '{dataset_dir}' remains untouched.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blur people and license plates in the dataset for public release.")
    parser.add_argument("--dir", "-d", type=str, default="dataset", help="Original private dataset directory to copy from")
    parser.add_argument("--out-dir", "-o", type=str, default="dataset_public", help="Output directory where the blurred images will exist")
    parser.add_argument("--conf", "-c", type=float, default=0.25, help="Confidence threshold for YOLOv8n object detection")
    
    args = parser.parse_args()
    anonymize_dataset(args.dir, args.out_dir, args.conf)
