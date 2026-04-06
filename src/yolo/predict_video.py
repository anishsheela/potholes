import cv2
import argparse
import os
import glob
from ultralytics import YOLO

def _process_single_video(video_path, model, output_dir):
    """
    Runs a custom YOLOv8 model on a single video file and saves the output.
    """
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
            # conf=0.5 means only draw boxes/masks if the AI is >50% sure it's a pothole
            results = model.predict(frame, conf=0.5, verbose=False)
            
            # The result object has a plot() method that draws the boxes AND masks directly on the frame
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
    print(f"Potholes Detected: {potholes_detected} (Total detections across all frames)")
    print(f"Saved Annotated Video to: {output_path}")

def predict_video(input_path, model_weights, output_dir):
    """
    Runs a custom YOLOv8 model on a video file or directory of videos and saves the output.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input path not found: {input_path}")
        return
        
    if not os.path.exists(model_weights):
        print(f"Error: Model weights not found: {model_weights}")
        print("Please run train_yolo.py first to generate the best.pt file.")
        return

    print(f"Loading custom YOLO model from {model_weights}...")
    import torch
    if torch.cuda.is_available():
        device = 'cuda:0'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Using hardware acceleration: {device}")
    model = YOLO(model_weights)
    model.to(device)

    if os.path.isdir(input_path):
        video_files = []
        for ext in ('*.mp4', '*.MP4', '*.avi', '*.AVI', '*.mov', '*.MOV'):
            video_files.extend(glob.glob(os.path.join(input_path, ext)))
        if not video_files:
            print(f"No video files found in directory: {input_path}")
            return
        video_paths = sorted(video_files)
        print(f"Found {len(video_paths)} videos in {input_path}. Processing...")
    else:
        video_paths = [input_path]

    for v_path in video_paths:
        _process_single_video(v_path, model, output_dir)
        print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pothole Detection Video Inference")
    # We keep --video for backward compatibility, but it also accepts directories now
    parser.add_argument("--video", "-v", "--source", "-s", type=str, required=True, dest="source", help="Path to input video or directory containing videos")
    parser.add_argument("--weights", "-w", type=str, default="models/best_pothole.pt", help="Path to trained YOLO model .pt file")
    parser.add_argument("--out-dir", "-o", type=str, default="processed_data/annotated_videos", help="Directory to save the final video")
    
    args = parser.parse_args()
    
    predict_video(args.source, args.weights, args.out_dir)
