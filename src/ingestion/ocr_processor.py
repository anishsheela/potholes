import cv2
import pytesseract
import os
import argparse

# On NixOS the tesseract binary lives in the nix store; the shellHook
# exports TESSERACT_CMD so subprocesses can find it regardless of PATH.
if 'TESSERACT_CMD' in os.environ:
    pytesseract.pytesseract.tesseract_cmd = os.environ['TESSERACT_CMD']

import glob
import pandas as pd
import re
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Legacy OCR engine (oem 0) is 5-10x faster than LSTM (oem 3) for fixed
# dashcam fonts and gives equal or better accuracy on high-contrast crops.
_TESS_CONFIG = '--psm 7 --oem 0'


def _prepare_roi(crop):
    """
    Convert an ROI crop to a high-contrast grayscale image that the legacy
    Tesseract engine can read accurately.  Upscaling to at least 60px tall
    and applying Otsu binarisation are the two changes that most help.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # Upscale if the crop is too small (Tesseract accuracy degrades below ~30px)
    h = gray.shape[0]
    if h < 60:
        scale = 60 / h
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)
    # Otsu threshold → clean black-on-white binary image
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def extract_data_from_frame(args):
    """
    Worker function to process a single frame.
    Reads the image, crops exact text rectangles based on config, runs OCR,
    and returns the data dictionary.
    """
    frame_path, camera_config = args

    # Re-apply tesseract path in worker process (needed for spawn-based pools)
    if 'TESSERACT_CMD' in os.environ:
        pytesseract.pytesseract.tesseract_cmd = os.environ['TESSERACT_CMD']

    # Extract timestamp from filename (e.g., frame_0001.0s.jpg -> 1.0)
    filename = os.path.basename(frame_path)
    match = re.search(r"frame_(\d+\.\d+)s\.jpg", filename)
    video_sec = float(match.group(1)) if match else 0.0

    video_name = os.path.basename(os.path.dirname(frame_path))

    if not camera_config:
        return None

    img = cv2.imread(frame_path)
    if img is None:
        return None

    date_coords = camera_config['date_roi']['coords']   # [y1, y2, x1, x2]
    gps_coords  = camera_config['gps_roi']['coords']

    y1, y2, x1, x2 = date_coords
    date_time_box = _prepare_roi(img[y1:y2, x1:x2])

    y1, y2, x1, x2 = gps_coords
    gps_speed_box = _prepare_roi(img[y1:y2, x1:x2])

    dt_text      = pytesseract.image_to_string(date_time_box, config=_TESS_CONFIG).strip()
    gps_speed_str = pytesseract.image_to_string(gps_speed_box, config=_TESS_CONFIG).strip()

    # --- Parse Date & Time ---
    date_str = ""
    time_str = ""
    date_match = re.search(camera_config['date_roi']['regex'], dt_text, re.IGNORECASE)
    if date_match:
        try:
            date_str = date_match.group('date')
        except IndexError:
            date_str = date_match.group(1)

    for text in dt_text.split():
        if ":" in text and len(text) > 5:
            time_str = text

    # --- Parse Speed & GPS ---
    speed = None
    lat   = None
    lon   = None

    gps_speed_str_clean = gps_speed_str.strip().replace(" ", "")
    gps_match = re.search(camera_config['gps_roi']['regex'],
                          gps_speed_str_clean, re.IGNORECASE)

    if gps_match:
        grouped_dict = gps_match.groupdict()
        if 'speed' in grouped_dict: speed = grouped_dict['speed']
        if 'lat'   in grouped_dict: lat   = grouped_dict['lat']
        if 'lon'   in grouped_dict: lon   = grouped_dict['lon']

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
        "Video":        video_name,
        "Video_Seconds": video_sec,
        "Date":         date_str,
        "Time":         time_str,
        "Speed_kmh":    speed,
        "Latitude":     lat,
        "Longitude":    lon,
        "Raw_OCR":      gps_speed_str,
        "Frame_Path":   frame_path,
    }


def process_frames(frames_dir, output_csv, config_path, out_invalid=None,
                   max_workers=None):
    with open(config_path, 'r') as f:
        master_config = yaml.safe_load(f)

    processed_videos = set()
    existing_df = None

    if os.path.exists(output_csv):
        try:
            existing_df = pd.read_csv(output_csv)
            if 'Video' in existing_df.columns:
                processed_videos = set(existing_df['Video'].unique())
                print(f"Found existing data for {len(processed_videos)} videos.")
        except Exception as e:
            print(f"Failed to read existing CSV: {e}")

    all_frame_files = glob.glob(os.path.join(frames_dir, "**/*.jpg"), recursive=True)
    all_frame_files.sort()

    if not all_frame_files:
        print(f"No frames found in {frames_dir}")
        return

    new_frame_args = []
    for frame_path in all_frame_files:
        video_name  = os.path.basename(os.path.dirname(frame_path))
        camera_name = os.path.basename(os.path.dirname(os.path.dirname(frame_path)))

        if video_name not in processed_videos:
            camera_config = master_config['cameras'].get(camera_name)
            if not camera_config:
                print(f"Warning: unknown camera '{camera_name}', skipping.")
                continue
            new_frame_args.append((frame_path, camera_config))

    if not new_frame_args:
        print("All frames already processed. Nothing to do.")
        return

    # Cap workers: beyond ~8-12 the per-process tesseract subprocess overhead
    # causes thrashing.  On a high-core server uncapped workers hurt throughput.
    cpu_count = os.cpu_count() or 4
    workers = max_workers or min(cpu_count, 12)
    print(f"Found {len(new_frame_args):,} NEW frames to process.")
    print(f"Starting OCR with {workers} workers (oem=legacy, psm=7)...")

    new_results_data = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(extract_data_from_frame, a): a
                   for a in new_frame_args}

        with tqdm(total=len(new_frame_args), desc="Running OCR", unit="frame") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        new_results_data.append(result)
                except Exception as exc:
                    print(f"Frame error: {exc}")
                finally:
                    pbar.update(1)

    if not new_results_data:
        print("No valid data extracted from new frames.")
        return

    new_df = pd.DataFrame(new_results_data)

    invalid_gps_df = new_df[new_df['Latitude'].isna() | new_df['Longitude'].isna()].copy()

    no_gps_signal = invalid_gps_df[
        invalid_gps_df['Raw_OCR'].str.contains(r'--+|—+|^\s*$', regex=True, na=True)
    ]
    ocr_failures = invalid_gps_df[
        ~invalid_gps_df['Raw_OCR'].str.contains(r'--+|—+|^\s*$', regex=True, na=True)
    ]

    initial_len = len(new_df)
    new_df = new_df.dropna(subset=['Latitude', 'Longitude'])

    print(f"\n{'='*60}")
    print(f"GPS EXTRACTION SUMMARY:")
    print(f"{'='*60}")
    print(f"Total frames processed:           {initial_len:,}")
    print(f"Valid GPS coordinates:            {len(new_df):,} ({len(new_df)/initial_len*100:.1f}%)")
    print(f"\nDiscarded frames:                 {initial_len - len(new_df):,} ({(initial_len - len(new_df))/initial_len*100:.1f}%)")
    print(f"  ├─ No GPS signal (camera):      {len(no_gps_signal):,} ({len(no_gps_signal)/initial_len*100:.1f}%)")
    print(f"  └─ OCR failed to extract:       {len(ocr_failures):,} ({len(ocr_failures)/initial_len*100:.1f}%)")
    print(f"{'='*60}\n")

    if out_invalid and not invalid_gps_df.empty:
        os.makedirs(os.path.dirname(out_invalid), exist_ok=True)
        invalid_gps_df[['Video', 'Video_Seconds', 'Frame_Path', 'Raw_OCR']].to_csv(
            out_invalid, index=False)
        print(f"Saved {len(invalid_gps_df)} invalid-GPS rows to {out_invalid}")

    if existing_df is not None and not existing_df.empty:
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df

    combined_df = combined_df.sort_values(
        by=['Video', 'Video_Seconds']).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    combined_df.to_csv(output_csv, index=False)
    print(f"Appended {len(new_df):,} rows. Total saved to {output_csv}: {len(combined_df):,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera",      type=str, required=True)
    parser.add_argument("--config", "-c", type=str, default="config.yaml")
    parser.add_argument("--workers",     type=int, default=None,
                        help="Number of parallel workers (default: min(cpu_count, 12))")
    args = parser.parse_args()

    input_dir   = os.path.join("processed_data", "frames", args.camera)
    output_csv  = os.path.join("processed_data", "route_data", f"{args.camera}.csv")
    out_invalid = os.path.join("processed_data", "route_data", f"invalid_gps_{args.camera}.csv")

    process_frames(input_dir, output_csv, args.config, out_invalid,
                   max_workers=args.workers)
