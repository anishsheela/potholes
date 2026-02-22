import cv2
import pytesseract
import argparse
import os
import glob
import pandas as pd
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def extract_data_from_frame(frame_path):
    """
    Worker function to process a single frame.
    Reads the image, crops exact text rectangles, runs OCR, and returns the data dictionary.
    """
    
    # Extract timestamp from filename (e.g., frame_0001.0s.jpg -> 1.0)
    filename = os.path.basename(frame_path)
    match = re.search(r"frame_(\d+\.\d+)s\.jpg", filename)
    video_sec = float(match.group(1)) if match else 0.0
    
    # Determine the video name for groupings
    video_name = os.path.basename(os.path.dirname(frame_path))
    
    # Read image
    img = cv2.imread(frame_path)
    if img is None:
        return None
        
    height, width, _ = img.shape
    
    # User provided EXACT pixel coordinates
    # Top Left (26,1858) Bottom Right (753,1934) - Date/Time
    # Top Left (1500,1848) Bottom Right (2404,1935) - Speed/GPS
    
    # Numpy slicing is [y1:y2, x1:x2]
    # 1. Date/Time ROI (Left side)
    date_time_box = img[1858:1934, 26:753]
    
    # 2. GPS/Speed ROI (Right side)
    gps_speed_box = img[1848:1935, 1500:2404]
    
    # Run OCR independently on both tiny, high-contrast boxes
    # psm 7 implies single line of text
    dt_text = pytesseract.image_to_string(date_time_box, config='--psm 7').strip()
    gps_speed_str = pytesseract.image_to_string(gps_speed_box, config='--psm 7').strip()
    
    # Parse Left Side (Date & Time)
    date_str = ""
    time_str = ""
    parts = dt_text.split()
    for text in parts:
        if "-" in text and len(text) > 8:
            date_str = text
        elif ":" in text and len(text) > 5:
            time_str = text
        
    speed = None
    lat = None
    lon = None
    
    gps_speed_str = gps_speed_str.strip().replace(" ", "")
    
    speed_match = re.search(r"(\d+)km/h", gps_speed_str, re.IGNORECASE)
    if speed_match:
        speed = speed_match.group(1)
        
    lon_match = re.search(r"[EW](\d+\.\d+)", gps_speed_str, re.IGNORECASE)
    lat_match = re.search(r"[NS](\d+\.\d+)", gps_speed_str, re.IGNORECASE)
    
    if lon_match:
        lon = lon_match.group(1)
    if lat_match:
        lat = lat_match.group(1)
        
    return {
        "Video": video_name,
        "Video_Seconds": video_sec,
        "Date": date_str,
        "Time": time_str,
        "Speed_kmh": speed,
        "Latitude": lat,
        "Longitude": lon,
        "Raw_OCR": gps_speed_str
    }

def process_frames(frames_dir, output_csv):
    """
    Discovers new frames and distributes the OCR workload using ProcessPoolExecutor.
    If output_csv exists, it skips videos that have already been processed and appends new data.
    """
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
        
    # 3. Filter out frames from already processed videos
    new_frame_files = []
    for frame_path in all_frame_files:
        video_name = os.path.basename(os.path.dirname(frame_path))
        if video_name not in processed_videos:
            new_frame_files.append(frame_path)
            
    if not new_frame_files:
        print(f"All {len(all_frame_files)} frames belong to already processed videos. Skipping OCR.")
        return

    print(f"Found {len(new_frame_files)} NEW frames to process.")
    print(f"Starting Multi-Core OCR Processing...")
    
    new_results_data = []

    # 4. Use multi-processing to run CPU-bound OCR in parallel on new frames only
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(extract_data_from_frame, path): path for path in new_frame_files}
        
        with tqdm(total=len(new_frame_files), desc="Running OCR", unit="frame") as pbar:
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
    
    # Filter out empty GPS points from new data
    initial_len = len(new_df)
    new_df = new_df.dropna(subset=['Latitude', 'Longitude'])
    print(f"\nFiltered {initial_len - len(new_df)} new frames missing valid GPS coordinates.")
    
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
    parser.add_argument("--input", "-i", type=str, default="output/frames", help="Directory containing extracted frames")
    parser.add_argument("--output", "-o", type=str, default="output/route_data.csv", help="Output CSV file path")
    args = parser.parse_args()
    
    process_frames(args.input, args.output)
