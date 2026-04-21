import pandas as pd
import argparse
import os
import shutil
import time
import math
from geopy.distance import geodesic

# ==========================================
# CONSTANTS & CONFIGURATION
# ==========================================
MIN_SPEED_KMH = 5.0
DEFAULT_MAX_SPEED_KMH = 150.0
# ==========================================


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


def filter_frames(csv_path, frames_dir, output_csv, output_dir,
                  max_speed_kmh=DEFAULT_MAX_SPEED_KMH,
                  args_out_invalid=None,
                  disable_idle=False,
                  disable_anomaly=False,
                  missing_log=None):

    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    print(f"Reading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Parse Lat/Lon
    df['Lat_Float'] = df['Latitude'].apply(safe_float)
    df['Lon_Float'] = df['Longitude'].apply(safe_float)

    initial_len = len(df)

    invalid_gps_df = df[df['Lat_Float'].isna() | df['Lon_Float'].isna()].copy()
    df = df.dropna(subset=['Lat_Float', 'Lon_Float'])
    print(f"Dropped {initial_len - len(df)} rows with unparseable coordinates.")

    if args_out_invalid and not invalid_gps_df.empty:
        os.makedirs(os.path.dirname(args_out_invalid), exist_ok=True)
        invalid_gps_df.drop(columns=['Lat_Float', 'Lon_Float']).to_csv(
            args_out_invalid, index=False)
        print(f"Saved {len(invalid_gps_df)} invalid-GPS rows to {args_out_invalid}")

    if df.empty:
        print("Error: No valid GPS data to process.")
        return

    # Filter idle frames
    idle_dropped = 0
    if not disable_idle:
        df['Speed_Float'] = pd.to_numeric(df['Speed_kmh'], errors='coerce').fillna(0)
        filtered_df = df[df['Speed_Float'] > MIN_SPEED_KMH].copy()
        idle_dropped = len(df) - len(filtered_df)
        print(f"Dropped {idle_dropped} idle frames (speed < {MIN_SPEED_KMH} km/h).")
    else:
        print("Skipping idle filtering...")
        filtered_df = df.copy()

    filtered_df = filtered_df.sort_values(
        by=['Video', 'Video_Seconds']).reset_index(drop=True)

    print(f"Checking GPS anomalies (max logical speed: {max_speed_kmh} km/h)...")

    kept_indices = []
    anomaly_dropped = 0

    current_video = None
    anchor_coords = None
    anchor_sec = None

    start_time = time.time()

    for i, row in filtered_df.iterrows():
        video    = row['Video']
        video_sec = row['Video_Seconds']
        coords   = (row['Lat_Float'], row['Lon_Float'])

        if video != current_video:
            current_video  = video
            anchor_coords  = coords
            anchor_sec     = video_sec
            kept_indices.append(i)
            continue

        time_diff_sec = video_sec - anchor_sec
        if time_diff_sec <= 0:
            kept_indices.append(i)
            continue

        distance_m       = geodesic(anchor_coords, coords).meters
        implied_speed    = (distance_m / time_diff_sec) * 3.6

        if disable_anomaly or implied_speed <= max_speed_kmh:
            kept_indices.append(i)
            anchor_coords = coords
            anchor_sec    = video_sec
        else:
            anomaly_dropped += 1

    final_df = filtered_df.iloc[kept_indices].copy()

    drop_cols = ['Lat_Float', 'Lon_Float', 'Speed_Float', 'Is_Daylight']
    final_df = final_df.drop(columns=drop_cols, errors='ignore')

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    final_df.to_csv(output_csv, index=False)

    print(f"\n--- Processing Summary ---")
    print(f"Original Frames : {initial_len:,}")
    print(f"Dropped Idle    : {idle_dropped:,}")
    print(f"Dropped Anomaly : {anomaly_dropped:,}")
    print(f"Kept Frames     : {len(final_df):,}")
    print(f"Saved to {output_csv}")

    # Copy physical frames
    if frames_dir and output_dir:
        print(f"\nCopying frames to {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)
        copied = 0
        missing = 0
        missing_rows = []

        for _, row in final_df.iterrows():
            filename = format_filename(row['Video_Seconds'])
            src = os.path.join(frames_dir, row['Video'], filename)

            if os.path.exists(src):
                dest_folder = os.path.join(output_dir, row['Video'])
                os.makedirs(dest_folder, exist_ok=True)
                shutil.copy2(src, os.path.join(dest_folder, filename))
                copied += 1
            else:
                missing += 1
                missing_rows.append({
                    'expected_path': src,
                    'video':         row['Video'],
                    'seconds':       row['Video_Seconds'],
                })

            if copied % 500 == 0 and copied > 0:
                print(f"  Copied {copied}/{len(final_df)} frames...")

        print(f"Done. {copied:,} copied, {missing:,} missing.")

        if missing_log and missing_rows:
            os.makedirs(os.path.dirname(missing_log) or '.', exist_ok=True)
            pd.DataFrame(missing_rows).to_csv(missing_log, index=False)
            print(f"Missing file paths written to {missing_log}")

    elapsed = time.time() - start_time
    print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter OCR output by speed/anomaly and copy valid frames.")
    parser.add_argument("--camera",         type=str, required=True)
    parser.add_argument("--max-speed",      type=float, default=DEFAULT_MAX_SPEED_KMH)
    parser.add_argument("--disable-idle",   action="store_true")
    parser.add_argument("--disable-anomaly", action="store_true")
    parser.add_argument("--missing-log",    type=str, default=None,
                        help="CSV path to write missing frame paths for debugging")
    args = parser.parse_args()

    csv_path    = os.path.join("processed_data", "route_data", f"{args.camera}.csv")
    frames_dir  = os.path.join("processed_data", "frames", args.camera)
    out_csv     = os.path.join("processed_data", "route_data", f"filtered_{args.camera}.csv")
    out_dir     = os.path.join("processed_data", "filtered_frames", args.camera)
    out_invalid = os.path.join("processed_data", "route_data",
                               f"invalid_gps_{args.camera}.csv")

    filter_frames(
        csv_path, frames_dir, out_csv, out_dir,
        max_speed_kmh=args.max_speed,
        args_out_invalid=out_invalid,
        disable_idle=args.disable_idle,
        disable_anomaly=args.disable_anomaly,
        missing_log=args.missing_log,
    )
