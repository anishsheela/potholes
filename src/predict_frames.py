import cv2
import argparse
import os
import glob
from ultralytics import YOLO
from tqdm import tqdm

def predict_frames(input_dir, model_weights, output_dir):
    """
    Runs a custom YOLOv8 model on a directory of images and saves the annotated frames.
    """
    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        return
        
    if not os.path.exists(model_weights):
        print(f"Error: Model weights not found: {model_weights}")
        print("Please run train_yolo.py first to generate the best.pt file.")
        return

    print(f"Loading custom YOLO model from {model_weights}...")
    model = YOLO(model_weights)
    
    # Get all .jpg files in the directory
    image_paths = glob.glob(os.path.join(input_dir, "*.jpg")) + glob.glob(os.path.join(input_dir, "*.png"))
    image_paths.sort() # Ensure chronological order if filenames are sequential
    
    if not image_paths:
        print(f"No images found in {input_dir}")
        return
        
    # Create the output directory based on the input folder's name
    folder_name = os.path.basename(os.path.normpath(input_dir))
    final_output_dir = os.path.join(output_dir, f"{folder_name}_annotated")
    os.makedirs(final_output_dir, exist_ok=True)
    
    print(f"Processing {len(image_paths)} images from {input_dir}...")
    
    potholes_detected = 0
    
    for img_path in tqdm(image_paths, desc="Inferencing", unit="frame"):
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        # Run YOLO inference
        # conf=0.5 means only draw boxes if the AI is >50% sure it's a pothole
        results = model.predict(img, conf=0.5, verbose=False)
        
        # The result object has a plot() method that draws the boxes directly on the frame
        annotated_frame = results[0].plot()
        
        potholes_in_frame = len(results[0].boxes)
        potholes_detected += potholes_in_frame
        
        # Save the annotated frame ONLY if a pothole was detected
        if potholes_in_frame > 0:
            filename = os.path.basename(img_path)
            out_path = os.path.join(final_output_dir, filename)
            cv2.imwrite(out_path, annotated_frame)
        
    print("\n--- Inference Complete ---")
    print(f"Frames Processed:  {len(image_paths)}")
    print(f"Potholes Detected: {potholes_detected} (Total boxes drawn across all frames)")
    print(f"Saved Annotated Frames to: {final_output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pothole Detection Frame Inference")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input directory of frames")
    parser.add_argument("--weights", "-w", type=str, default="runs/detect/runs/detect/train/weights/best.pt", help="Path to trained YOLO model .pt file")
    parser.add_argument("--out-dir", "-o", type=str, default="output/annotated_frames", help="Directory to save the final images")
    
    args = parser.parse_args()
    
    predict_frames(args.input, args.weights, args.out_dir)
