import cv2
import argparse
import os
from ultralytics import YOLO

def predict_video(video_path, model_weights, output_dir):
    """
    Runs a custom YOLOv8 model on a video file and saves the output.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return
        
    if not os.path.exists(model_weights):
        print(f"Error: Model weights not found: {model_weights}")
        print("Please run train_yolo.py first to generate the best.pt file.")
        return

    print(f"Loading custom YOLO model from {model_weights}...")
    model = YOLO(model_weights)
    
    video_name = os.path.basename(video_path)
    base_name, ext = os.path.splitext(video_name)
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"{base_name}_annotated{ext}")
    print(f"Processing video {video_name}...")
    
    # Open the video cap
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video {video_path}")
        return
        
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize the VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frames_processed = 0
    potholes_detected = 0
    
    from tqdm import tqdm
    with tqdm(total=total_frames, desc="Inferencing", unit="frame") as pbar:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            # Run YOLO inference
            # conf=0.5 means only draw boxes if the AI is >50% sure it's a pothole
            results = model.predict(frame, conf=0.5, verbose=False)
            
            # The result object has a plot() method that draws the boxes directly on the frame
            annotated_frame = results[0].plot()
            
            potholes_in_frame = len(results[0].boxes)
            potholes_detected += potholes_in_frame
            
            out.write(annotated_frame)
            frames_processed += 1
            pbar.update(1)
            
    cap.release()
    out.release()
    
    print("\n--- Inference Complete ---")
    print(f"Frames Processed:  {frames_processed}")
    print(f"Potholes Detected: {potholes_detected} (Total boxes drawn across all frames)")
    print(f"Saved Annotated Video to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pothole Detection Video Inference")
    parser.add_argument("--video", "-v", type=str, required=True, help="Path to input .MP4 video")
    parser.add_argument("--weights", "-w", type=str, default="runs/detect/runs/detect/train/weights/best.pt", help="Path to trained YOLO model .pt file")
    parser.add_argument("--out-dir", "-o", type=str, default="output/annotated_videos", help="Directory to save the final video")
    
    args = parser.parse_args()
    
    predict_video(args.video, args.weights, args.out_dir)
