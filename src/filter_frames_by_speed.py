import pandas as pd
import argparse
import os
import shutil
import time
from geopy.distance import geodesic

def safe_float(val):
    try:
        clean_val = ''.join(c for c in str(val) if c.isdigit() or c == '.')
        parts = clean_val.split('.')
        if len(parts) > 2:
            clean_val = parts[0] + '.' + ''.join(parts[1:])
        return float(clean_val)
    except ValueError:
        return None

def format_filename(video_sec):
    return f"frame_{video_sec:06.1f}s.jpg"

def filter_frames(csv_path, frames_dir, output_csv, output_dir, max_speed_kmh=150.0):
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    print(f"Reading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # 1. Parse Lat/Lon gracefully
    df['Lat_Float'] = df['Latitude'].apply(safe_float)
    df['Lon_Float'] = df['Longitude'].apply(safe_float)

    initial_len = len(df)
    df = df.dropna(subset=['Lat_Float', 'Lon_Float'])
    print(f"Dropped {initial_len - len(df)} rows with unparseable coordinates.")
    
    if df.empty:
        print("Error: No valid GPS data to process.")
        return

    # 2. Filter out nighttime frames (e.g., before 06:00 or after 18:00)
    print("Filtering out nighttime frames...")
    
    def is_daylight(time_str):
        if not isinstance(time_str, str):
            return False
        try:
            # Parse HH:MM:SS
            parts = time_str.split(':')
            if len(parts) >= 2:
                hour = int(parts[0].strip())
                # Keep frames between 06:00 and 17:59
                return 6 <= hour < 18
            return False
        except:
            return False
            
    df['Is_Daylight'] = df['Time'].apply(is_daylight)
    daylight_df = df[df['Is_Daylight']].copy()
    night_dropped = len(df) - len(daylight_df)
    print(f"Dropped {night_dropped} frames captured outside of daylight hours (06:00 - 18:00).")

    # 3. Filter out idle frames (Speed == 0)
    daylight_df['Speed_Float'] = pd.to_numeric(daylight_df['Speed_kmh'], errors='coerce').fillna(0)
    filtered_df = daylight_df[daylight_df['Speed_Float'] > 0].copy()
    idle_dropped = len(daylight_df) - len(filtered_df)
    print(f"Dropped {idle_dropped} frames where speed is 0 km/h.")

    # Sort chronologically by Video and Time
    filtered_df = filtered_df.sort_values(by=['Video', 'Video_Seconds']).reset_index(drop=True)

    print(f"Filtering OCR anomalies (Max logical speed: {max_speed_kmh} km/h)...")
    
    kept_indices = []
    anomaly_dropped = 0
    current_video = None
    anchor_coords = None
    anchor_sec = None

    start_time = time.time()

    # 3. Anomaly detection (Impossible Jumps)
    for i, row in filtered_df.iterrows():
        video = row['Video']
        video_sec = row['Video_Seconds']
        coords = (row['Lat_Float'], row['Lon_Float'])
        
        # New valid group
        if video != current_video:
            current_video = video
            anchor_coords = coords
            anchor_sec = video_sec
            kept_indices.append(i)
            continue
            
        time_diff_sec = video_sec - anchor_sec
        if time_diff_sec <= 0:
            # Duplicate time stamp or weird sorting, just keep it to be safe
            kept_indices.append(i)
            continue
            
        # Physical Euclidean distance
        distance_m = geodesic(anchor_coords, coords).meters
        
        # Implied speed in km/h
        implied_speed_kmh = (distance_m / time_diff_sec) * 3.6
        
        # If the jump is physically possible, this new point becomes the anchor.
        # This handles cases where OCR temporarily glitches one digit to another country.
        if implied_speed_kmh <= max_speed_kmh:
            kept_indices.append(i)
            anchor_coords = coords
            anchor_sec = video_sec
        else:
            anomaly_dropped += 1
            
    final_df = filtered_df.iloc[kept_indices].copy()
    
    # Clean up temporary parsing columns
    final_df = final_df.drop(columns=['Lat_Float', 'Lon_Float', 'Speed_Float'])
    
    # Save the new CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    final_df.to_csv(output_csv, index=False)
    
    print(f"\n--- Filtering Summary ---")
    print(f"Original Frames: {initial_len}")
    print(f"Dropped Idle:    {idle_dropped}")
    print(f"Dropped Anomaly: {anomaly_dropped}")
    print(f"Kept Frames:     {len(final_df)}")
    print(f"Saved filtered data to {output_csv}")
    
    # physical copies
    if frames_dir and output_dir:
        print(f"\nCopying kept frames to {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)
        copied = 0
        missing = 0
        
        for _, row in final_df.iterrows():
            video_name = row['Video']
            video_sec = row['Video_Seconds']
            filename = format_filename(video_sec)
            
            src_path = os.path.join(frames_dir, video_name, filename)
            
            if os.path.exists(src_path):
                dest_folder = os.path.join(output_dir, video_name)
                os.makedirs(dest_folder, exist_ok=True)
                dest_path = os.path.join(dest_folder, filename)
                shutil.copy2(src_path, dest_path)
                copied += 1
            else:
                missing += 1
                
            if copied % 100 == 0 and copied > 0:
                print(f"Copied {copied}/{len(final_df)} frames...")
                
        print(f"Finished copying. {copied} successful, {missing} missing files.")
        
    elapsed = time.time() - start_time
    print(f"Total processing time: {elapsed:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter redundant frames and OCR anomalies.")
    parser.add_argument("--csv", type=str, default="output/route_data.csv", help="Input OCR CSV")
    parser.add_argument("--max-speed", type=float, default=150.0, help="Max valid speed (km/h) between frames to prevent GPS jumps")
    parser.add_argument("--frames-dir", type=str, default="output/frames", help="Original directory")
    parser.add_argument("--out-csv", type=str, default="output/filtered_route_data.csv", help="Filtered CSV")
    parser.add_argument("--out-dir", type=str, default="output/filtered_frames", help="Filtered image directory")
    
    args = parser.parse_args()
    
    filter_frames(args.csv, args.frames_dir, args.out_csv, args.out_dir, args.max_speed)
