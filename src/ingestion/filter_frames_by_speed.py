import pandas as pd
import argparse
import os
import shutil
import time
import math
import pygeohash as pgh
from geopy.distance import geodesic

# ==========================================
# CONSTANTS & CONFIGURATION
# ==========================================
# Time filtering (24-hour format)
DAYLIGHT_START_HOUR = 6
DAYLIGHT_END_HOUR = 18

# Speed/Idle filtering
MIN_SPEED_KMH = 5.0

# Anomaly detection (GPS jump filtering)
DEFAULT_MAX_SPEED_KMH = 150.0

# Spatial Deduplication
GEOHASH_PRECISION = 8  # Precision 8 is approx 38m x 19m
# ==========================================

def calculate_bearing(pointA, pointB):
    """Calculates the bearing between two GPS points."""
    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])
    diffLong = math.radians(pointB[1] - pointA[1])
    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(diffLong))
    return (math.degrees(math.atan2(x, y)) + 360) % 360

def bearing_diff(b1, b2):
    """Calculates the absolute minimum difference between two bearings."""
    diff = abs(b1 - b2)
    return min(diff, 360 - diff)

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

def filter_frames(csv_path, frames_dir, output_csv, output_dir, max_speed_kmh=DEFAULT_MAX_SPEED_KMH, 
                  args_out_invalid=None, disable_nighttime=False, disable_idle=False, 
                  disable_anomaly=False, disable_spatial=False):
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    print(f"Reading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # 1. Parse Lat/Lon gracefully
    df['Lat_Float'] = df['Latitude'].apply(safe_float)
    df['Lon_Float'] = df['Longitude'].apply(safe_float)

    initial_len = len(df)
    
    # Store the invalid GPS rows before dropping them
    invalid_gps_df = df[df['Lat_Float'].isna() | df['Lon_Float'].isna()].copy()
    
    df = df.dropna(subset=['Lat_Float', 'Lon_Float'])
    print(f"Dropped {initial_len - len(df)} rows with unparseable coordinates.")
    
    if args_out_invalid and not invalid_gps_df.empty:
        os.makedirs(os.path.dirname(args_out_invalid), exist_ok=True)
        # Drop the temporary parsing columns before saving
        invalid_gps_df = invalid_gps_df.drop(columns=['Lat_Float', 'Lon_Float'])
        invalid_gps_df.to_csv(args_out_invalid, index=False)
        print(f"Saved {len(invalid_gps_df)} rows with invalid GPS data to {args_out_invalid}")
    
    if df.empty:
        print("Error: No valid GPS data to process.")
        return

    # 2. Filter out nighttime frames (e.g., before 06:00 or after 18:00)
    night_dropped = 0
    if not disable_nighttime:
        print("Filtering out nighttime frames...")
        
        def is_daylight(time_str):
            if not isinstance(time_str, str):
                return False
            try:
                # Parse HH:MM:SS
                parts = time_str.split(':')
                if len(parts) >= 2:
                    hour = int(parts[0].strip())
                    # Keep frames between DAYLIGHT_START_HOUR and DAYLIGHT_END_HOUR
                    return DAYLIGHT_START_HOUR <= hour < DAYLIGHT_END_HOUR
                return False
            except:
                return False
                
        df['Is_Daylight'] = df['Time'].apply(is_daylight)
        daylight_df = df[df['Is_Daylight']].copy()
        night_dropped = len(df) - len(daylight_df)
        print(f"Dropped {night_dropped} frames captured outside of daylight hours ({DAYLIGHT_START_HOUR}:00 - {DAYLIGHT_END_HOUR}:00).")
    else:
        print("Skipping nighttime filtering...")
        daylight_df = df.copy()

    # 3. Filter out idle frames (Speed == 0)
    idle_dropped = 0
    if not disable_idle:
        daylight_df['Speed_Float'] = pd.to_numeric(daylight_df['Speed_kmh'], errors='coerce').fillna(0)
        filtered_df = daylight_df[daylight_df['Speed_Float'] > MIN_SPEED_KMH].copy()
        idle_dropped = len(daylight_df) - len(filtered_df)
        print(f"Dropped {idle_dropped} frames where speed is less than {MIN_SPEED_KMH} km/h.")
    else:
        print("Skipping idle filtering...")
        filtered_df = daylight_df.copy()

    # Sort chronologically by Video and Time
    filtered_df = filtered_df.sort_values(by=['Video', 'Video_Seconds']).reset_index(drop=True)

    print(f"Filtering OCR anomalies (Max logical speed: {max_speed_kmh} km/h) and deduplicating by Geohash...")
    
    # 3. Load Per-Camera Visited Geohashes
    # We want to deduplicate frames that fall on the exact same ~38m stretch of road
    # but ONLY against this specific camera's history.
    history_file = os.path.join(os.path.dirname(output_csv), f"visited_geohashes_{os.path.splitext(os.path.basename(output_csv))[0]}.txt")
    visited_hashes = set()
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            for line in f:
                visited_hashes.add(line.strip())
    print(f"Loaded {len(visited_hashes)} previously visited spatial zones for this camera.")
    
    kept_indices = []
    anomaly_dropped = 0
    spatial_dropped = 0
    
    current_video = None
    anchor_coords = None
    anchor_sec = None

    start_time = time.time()

    # 4. Anomaly detection & Spatial Deduplication (Turn Filtering removed)
    for i, row in filtered_df.iterrows():
        video = row['Video']
        video_sec = row['Video_Seconds']
        coords = (row['Lat_Float'], row['Lon_Float'])
        
        # Calculate Geohash (Precision 8 is approx 38m x 19m)
        current_hash = pgh.encode(coords[0], coords[1], precision=GEOHASH_PRECISION)
        
        if not disable_spatial and current_hash in visited_hashes:
            spatial_dropped += 1
            continue
        
        # New valid group
        if video != current_video:
            current_video = video
            anchor_coords = coords
            anchor_sec = video_sec
            kept_indices.append(i)
            visited_hashes.add(current_hash)
            continue
            
        time_diff_sec = video_sec - anchor_sec
        if time_diff_sec <= 0:
            # Duplicate time stamp or weird sorting, just keep it to be safe
            kept_indices.append(i)
            visited_hashes.add(current_hash)
            continue
            
        # Physical Euclidean distance
        distance_m = geodesic(anchor_coords, coords).meters
        
        # Implied speed in km/h
        implied_speed_kmh = (distance_m / time_diff_sec) * 3.6
        
        # If the jump is physically possible, this new point becomes the anchor.
        if disable_anomaly or implied_speed_kmh <= max_speed_kmh:
            kept_indices.append(i)
            visited_hashes.add(current_hash)
            anchor_coords = coords
            anchor_sec = video_sec
        else:
            anomaly_dropped += 1
            
    # Save the updated visited hashes history
    os.makedirs(os.path.dirname(history_file), exist_ok=True)
    with open(history_file, 'w') as f:
        for gh in visited_hashes:
            f.write(f"{gh}\n")
            
    final_df = filtered_df.iloc[kept_indices].copy()
    
    # Clean up temporary parsing columns
    drop_cols = ['Lat_Float', 'Lon_Float']
    if 'Speed_Float' in final_df.columns:
        drop_cols.append('Speed_Float')
    if 'Is_Daylight' in final_df.columns:
        drop_cols.append('Is_Daylight')
    final_df = final_df.drop(columns=drop_cols, errors='ignore')
    
    # Save the new CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    final_df.to_csv(output_csv, index=False)
    
    print(f"\n--- Filtering Summary ---")
    print(f"Original Frames:  {initial_len}")
    print(f"Dropped Night:    {night_dropped}")
    print(f"Dropped Idle:     {idle_dropped}")
    print(f"Dropped Anomaly:  {anomaly_dropped}")
    print(f"Dropped Redundant Road (Geohash): {spatial_dropped}")
    print(f"Kept Frames:      {len(final_df)}")
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
    parser.add_argument("--camera", type=str, required=True, help="Name of the camera folder (e.g. anish)")
    
    # Configuration overrides
    parser.add_argument("--max-speed", type=float, default=DEFAULT_MAX_SPEED_KMH, help="Max valid speed (km/h) between frames to prevent GPS jumps")
    
    # Disable flags
    parser.add_argument("--disable-nighttime", action="store_true", help="Disable filtering out frames captured at night (before 06:00 or after 18:00)")
    parser.add_argument("--disable-idle", action="store_true", help="Disable filtering out frames where speed is 0")
    parser.add_argument("--disable-anomaly", action="store_true", help="Disable detection and filtering of GPS jump anomalies")
    parser.add_argument("--disable-spatial", action="store_true", help="Disable deduplicating frames that fall in the same geohash")
    
    args = parser.parse_args()
    
    csv_path = os.path.join("processed_data", "route_data", f"{args.camera}.csv")
    frames_dir = os.path.join("processed_data", "frames", args.camera)
    out_csv = os.path.join("processed_data", "route_data", f"filtered_{args.camera}.csv")
    out_dir = os.path.join("processed_data", "filtered_frames", args.camera)
    out_invalid = os.path.join("processed_data", "route_data", f"invalid_gps_after_filter_{args.camera}.csv")
    
    filter_frames(
        csv_path, frames_dir, out_csv, out_dir, 
        max_speed_kmh=args.max_speed, 
        args_out_invalid=out_invalid,
        disable_nighttime=args.disable_nighttime,
        disable_idle=args.disable_idle,
        disable_anomaly=args.disable_anomaly,
        disable_spatial=args.disable_spatial
    )
