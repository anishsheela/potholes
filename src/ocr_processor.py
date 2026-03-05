import cv2
import pytesseract
import argparse
import os
import glob
import pandas as pd
import re
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def extract_data_from_frame(args):
    """
    Worker function to process a single frame.
    Reads the image, crops exact text rectangles based on config, runs OCR, and returns the data dictionary.
    """
    frame_path, camera_config = args
    
    # Extract timestamp from filename (e.g., frame_0001.0s.jpg -> 1.0)
    filename = os.path.basename(frame_path)
    match = re.search(r"frame_(\d+\.\d+)s\.jpg", filename)
    video_sec = float(match.group(1)) if match else 0.0
    
    # Determine the video name for groupings
    video_name = os.path.basename(os.path.dirname(frame_path))
    
    if not camera_config:
        print(f"Warning: No config found for camera folder. Skipping {frame_path}")
        return None
        
    # Read image
    img = cv2.imread(frame_path)
    if img is None:
        print(f"Failed to read image {frame_path}")
        return None
        
    height, width, _ = img.shape
    
    # Extract config parameters
    date_coords = camera_config['date_roi']['coords'] # [y1, y2, x1, x2]
    gps_coords = camera_config['gps_roi']['coords']
    
    # 1. Date/Time ROI
    y1, y2, x1, x2 = date_coords
    date_time_box = img[y1:y2, x1:x2]
    
    # 2. GPS/Speed ROI
    y1, y2, x1, x2 = gps_coords
    gps_speed_box = img[y1:y2, x1:x2]
    
    # Run OCR independently on both tiny, high-contrast boxes
    dt_text = pytesseract.image_to_string(date_time_box, config='--psm 7').strip()
    gps_speed_str = pytesseract.image_to_string(gps_speed_box, config='--psm 7').strip()
    
    # --- Parse left side (Date & Time) using Config Regex ---
    date_str = ""
    time_str = ""
    date_match = re.search(camera_config['date_roi']['regex'], dt_text, re.IGNORECASE)
    if date_match:
        # If the regex grouped date and time perfectly, use them. Or fallback to split.
        try:
            date_str = date_match.group('date')
        except IndexError:
            date_str = date_match.group(1) # fallback if group wasn't named
            
    # For time string, simple heuristic fallback if regex failed to capture it explicitly
    parts = dt_text.split()
    for text in parts:
        if ":" in text and len(text) > 5:
            time_str = text
            
    # --- Parse right side (Speed & GPS) using Config Regex ---
    speed = None
    lat = None
    lon = None
    
    gps_speed_str_clean = gps_speed_str.strip().replace(" ", "")
    gps_match = re.search(camera_config['gps_roi']['regex'], gps_speed_str_clean, re.IGNORECASE)
    
    if gps_match:
        grouped_dict = gps_match.groupdict()
        if 'speed' in grouped_dict: speed = grouped_dict['speed']
        if 'lat' in grouped_dict: lat = grouped_dict['lat']
        if 'lon' in grouped_dict: lon = grouped_dict['lon']
        
    # Fallback to old heuristic if regex didn't perfectly match
    if speed is None:
        speed_match = re.search(r"(\d+)km/h", gps_speed_str_clean, re.IGNORECASE)
        if speed_match:
            speed = speed_match.group(1)
        elif "--km/h" in gps_speed_str_clean.lower() or "---km/h" in gps_speed_str_clean.lower():
            speed = 0
            
    if lon is None:
        lon_match = re.search(r"[EW](\d+\.\d+)", gps_speed_str_clean, re.IGNORECASE)
        if lon_match: lon = lon_match.group(1)
    if lat is None:
        lat_match = re.search(r"[NS](\d+\.\d+)", gps_speed_str_clean, re.IGNORECASE)
        if lat_match: lat = lat_match.group(1)
        
    return {
        "Video": video_name,
        "Video_Seconds": video_sec,
        "Date": date_str,
        "Time": time_str,
        "Speed_kmh": speed,
        "Latitude": lat,
        "Longitude": lon,
        "Raw_OCR": gps_speed_str,
        "Frame_Path": frame_path
    }

def process_frames(frames_dir, output_csv, config_path, out_invalid=None):
    """
    Discovers new frames and distributes the OCR workload using ProcessPoolExecutor.
    If output_csv exists, it skips videos that have already been processed and appends new data.
    """
    with open(config_path, 'r') as f:
        master_config = yaml.safe_load(f)
        
    processed_videos = set()
    existing_df = None
    
    # 1. Check for existing data
    if os.path.exists(output_csv):
        try:
            existing_df = pd.read_csv(output_csv)
            if 'Video' in existing_df.columns:
                processed_videos = set(existing_df['Video'].unique())
                print(f"Found existing data for {len(processed_videos)} videos.")
        except Exception as e:
            print(f"Failed to read existing CSV: {e}")
            existing_df = None

    # 2. Grab all .jpg files from all subdirectories
    all_frame_files = glob.glob(os.path.join(frames_dir, "**/*.jpg"), recursive=True)
    all_frame_files.sort() # Ensure they are processed chronologically
    
    if not all_frame_files:
        print(f"No frames found in {frames_dir}")
        return
        
    # 4. Filter out frames from already processed videos and package args
    new_frame_args = []
    for frame_path in all_frame_files:
        video_name = os.path.basename(os.path.dirname(frame_path))
        # Path structure is processed_data/frames/anish/video1/frame.jpg
        camera_name = os.path.basename(os.path.dirname(os.path.dirname(frame_path)))
        
        if video_name not in processed_videos:
            camera_config = master_config['cameras'].get(camera_name)
            if not camera_config:
                print(f"Warning: Dropping frame from unknown camera folder '{camera_name}' (Not mapped in config.yaml).")
                continue
            new_frame_args.append((frame_path, camera_config))
            
    if not new_frame_args:
        print(f"All {len(all_frame_files)} frames belong to already processed videos, or are missing config maps. Skipping OCR.")
        return

    print(f"Found {len(new_frame_args)} NEW frames to process.")
    print(f"Starting Multi-Core Config-Driven OCR Processing...")
    
    new_results_data = []

    # 4. Use multi-processing to run CPU-bound OCR in parallel on new frames only
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(extract_data_from_frame, args): args for args in new_frame_args}
        
        with tqdm(total=len(new_frame_args), desc="Running OCR", unit="frame") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        new_results_data.append(result)
                except Exception as exc:
                    print(f"Frame processing generated an exception: {exc}")
                finally:
                    pbar.update(1)

    if not new_results_data:
        print("No valid data extracted from new frames.")
        return

    # 5. Save/Append to CSV
    new_df = pd.DataFrame(new_results_data)
    
    # Store the invalid GPS rows before dropping them
    invalid_gps_df = new_df[new_df['Latitude'].isna() | new_df['Longitude'].isna()].copy()
    
    # Filter out empty GPS points from new data
    initial_len = len(new_df)
    new_df = new_df.dropna(subset=['Latitude', 'Longitude'])
    print(f"\nFiltered {initial_len - len(new_df)} new frames missing valid GPS coordinates.")
    
    if out_invalid and not invalid_gps_df.empty:
        os.makedirs(os.path.dirname(out_invalid), exist_ok=True)
        # Option to only save relevant info for review
        invalid_gps_df[['Video', 'Video_Seconds', 'Frame_Path', 'Raw_OCR']].to_csv(out_invalid, index=False)
        print(f"Saved {len(invalid_gps_df)} rows with invalid GPS data to {out_invalid}")
    
    # Combine with existing data if present
    if existing_df is not None and not existing_df.empty:
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df
        
    # Final chronological sort
    combined_df = combined_df.sort_values(by=['Video', 'Video_Seconds']).reset_index(drop=True)
    
    # Ensure output dir exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # Save standard CSV
    combined_df.to_csv(output_csv, index=False)
    print(f"Appended {len(new_df)} new valid rows. Total records saved to {output_csv}: {len(combined_df)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=str, required=True, help="Name of the camera folder (e.g. anish)")
    parser.add_argument("--config", "-c", type=str, default="config.yaml", help="Path to camera mapping config.yaml")
    args = parser.parse_args()
    
    input_dir = os.path.join("processed_data", "frames", args.camera)
    output_csv = os.path.join("processed_data", "route_data", f"{args.camera}.csv")
    out_invalid = os.path.join("processed_data", "route_data", f"invalid_gps_{args.camera}.csv")
    
    process_frames(input_dir, output_csv, args.config, out_invalid)
