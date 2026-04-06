import os
import argparse
import glob
import subprocess
import time

def extract_frames_ffmpeg(video_path, output_dir, interval_sec=1):
    """
    Extracts frames from a video at a specified interval using FFmpeg.
    
    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save extracted frames.
        interval_sec (float): Interval in seconds between extracted frames.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    # Create output directory for this specific video
    video_filename = os.path.basename(video_path)
    video_name, _ = os.path.splitext(video_filename)
    video_out_dir = os.path.join(output_dir, video_name)
    
    # Check if frames already exist
    if os.path.exists(video_out_dir):
        existing_frames = glob.glob(os.path.join(video_out_dir, "*.jpg"))
        if len(existing_frames) > 0:
            print(f"Skipping '{video_filename}', already extracted {len(existing_frames)} frames.")
            return
            
    os.makedirs(video_out_dir, exist_ok=True)

    print(f"Processing '{video_filename}' using FFmpeg...")
    start_time = time.time()

    # Calculate FPS string for FFmpeg (e.g., 1/1 for 1 frame per second)
    fps_filter = f"fps=1/{interval_sec}"
    
    # Construct the output pattern, adding seconds timestamp to filename
    # We use %05d instead of seconds directly because FFmpeg doesn't easily template 
    # timestamps in filenames without complex metadata parsing.
    # However, since we extract exactly 1 frame per `interval_sec` seconds,
    # frame_00001.jpg corresponds to 0 seconds, frame_00002.jpg to 1 second, etc.
    # We will rename them after extraction to maintain compatibility with ocr_processor.py
    temp_output_pattern = os.path.join(video_out_dir, "temp_%05d.jpg")

    try:
        # Get duration for progress bar
        probe_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path]
        duration_str = subprocess.check_output(probe_cmd).decode('utf-8').strip()
        total_duration = float(duration_str)
        total_frames_expected = int(total_duration / interval_sec)
        
        # Command: ffmpeg -i video.mp4 -vf fps=1/1 -q:v 2 output_%05d.jpg
        cmd = [
            "ffmpeg", "-y", "-hwaccel", "auto", "-i", video_path,
            "-vf", fps_filter, "-q:v", "2", "-nostdin",
            "-progress", "-", "-nostats", # Output machine-readable progress to stdout
            temp_output_pattern
        ]

        # Run ffmpeg and parse progress
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True)
        
        from tqdm import tqdm
        with tqdm(total=total_frames_expected, desc=f"Extracting {video_filename}", unit="frame") as pbar:
            for line in process.stdout:
                if line.startswith("frame="):
                    try:
                        # Extract the frame number (e.g. "frame=123")
                        current_frame = int(line.split("=")[1].strip())
                        # Update progress bar by the difference
                        pbar.update(current_frame - pbar.n)
                    except ValueError:
                        pass
        
        process.wait()
        
        if process.returncode != 0:
            print(f"Error: FFmpeg exited with code {process.returncode}")
            return
            
        # Now rename the files to match the expected format (frame_0000.0s.jpg)
        temp_files = sorted(glob.glob(os.path.join(video_out_dir, "temp_*.jpg")))
        extracted_count = 0
        
        for i, temp_file in enumerate(temp_files):
            # Calculate the corresponding timestamp (0-indexed)
            timestamp_sec = i * interval_sec
            out_filename = f"frame_{timestamp_sec:06.1f}s.jpg"
            out_filepath = os.path.join(video_out_dir, out_filename)
            
            # Rename temp to final
            os.rename(temp_file, out_filepath)
            extracted_count += 1
            
        elapsed_time = time.time() - start_time
        print(f"Done! Extracted {extracted_count} frames to {video_out_dir} in {elapsed_time:.1f} seconds.\n")

    except subprocess.CalledProcessError as e:
        print(f"Error running FFmpeg on {video_filename}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(description="Extract full-resolution frames from dashcam videos using FFmpeg.")
    parser.add_argument("--input", "-i", type=str, default="videos", help="Input directory containing MP4 files OR path to a single MP4 file.")
    parser.add_argument("--output", "-o", type=str, default="output/frames", help="Output directory for saved frames.")
    parser.add_argument("--interval", type=float, default=1.0, help="Interval in seconds between extracted frames (default: 1.0).")
    
    args = parser.parse_args()

    # Ensure base output directory exists
    os.makedirs(args.output, exist_ok=True)

    # Check if ffmpeg is installed
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: FFmpeg is not installed or not found in system PATH.")
        print("Please install it using: brew install ffmpeg")
        return

    if os.path.isfile(args.input):
        print(f"Processing single file: {args.input}")
        extract_frames_ffmpeg(args.input, args.output, args.interval)
    elif os.path.isdir(args.input):
        print(f"Processing directory: {args.input}")
        # Find all MP4 files in the directory
        video_files = glob.glob(os.path.join(args.input, "*.MP4")) + glob.glob(os.path.join(args.input, "*.mp4"))
        
        if not video_files:
            print(f"No MP4 files found in {args.input}")
            return
            
        print(f"Found {len(video_files)} video files.")
        for video_file in sorted(video_files):
            extract_frames_ffmpeg(video_file, args.output, args.interval)
    else:
        print(f"Error: Invalid input path {args.input}")

if __name__ == "__main__":
    main()
