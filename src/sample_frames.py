import os
import random
import shutil
import argparse
import glob

def sample_frames(input_dir, output_dir, num_samples):
    # Find all jpegs in the input directory
    all_frames = glob.glob(os.path.join(input_dir, "**/*.jpg"), recursive=True)
    
    if not all_frames:
        print(f"Error: No images found in {input_dir}")
        return
        
    print(f"Found {len(all_frames)} total available frames.")
    
    # Check for already sampled frames (in staging or training) to prevent duplicates
    existing_sampled = set()
    training_dir = os.path.join(output_dir, "training")
    staging_dir = os.path.join(output_dir, "staging")
    
    # Ensure directories exist
    os.makedirs(training_dir, exist_ok=True)
    os.makedirs(staging_dir, exist_ok=True)
    
    for check_dir in [training_dir, staging_dir]:
        if os.path.exists(check_dir):
            existing_files = glob.glob(os.path.join(check_dir, "**/*.jpg"), recursive=True)
            for f in existing_files:
                existing_sampled.add(os.path.basename(f))
            
    print(f"Found {len(existing_sampled)} frames that have already been sampled previously.")
    
    # Filter out already sampled frames
    available_frames = []
    for frame_path in all_frames:
        video_name = os.path.basename(os.path.dirname(frame_path))
        filename = os.path.basename(frame_path)
        new_filename = f"{video_name}_{filename}"
        
        if new_filename not in existing_sampled:
            available_frames.append(frame_path)
            
    print(f"Available new frames to sample: {len(available_frames)}")
    
    if not available_frames:
        print("No new frames left to sample!")
        return
        
    # Cap the sample size if we requested more than we have
    samples_to_take = min(num_samples, len(available_frames))
    
    # Randomly select frames
    sampled_frames = random.sample(available_frames, samples_to_take)
    
    print(f"Randomly selected {samples_to_take} frames. Copying to {staging_dir}...")
    
    copied = 0
    for frame_path in sampled_frames:
        video_name = os.path.basename(os.path.dirname(frame_path))
        filename = os.path.basename(frame_path)
        
        new_filename = f"{video_name}_{filename}"
        dest_path = os.path.join(staging_dir, new_filename)
        
        shutil.copy2(frame_path, dest_path)
        copied += 1
        
        if copied % 50 == 0:
            print(f"Copied {copied}/{samples_to_take}...")
            
    print(f"\nSuccessfully copied {copied} new frames to {staging_dir}.")
    print("You can now run 'python src/generate_label_studio_tasks.py' to auto-annotate them!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Randomly sample frames for YOLO annotation.")
    parser.add_argument("--input", "-i", type=str, default="output/filtered_frames", help="Input directory containing video folders with frames")
    parser.add_argument("--output", "-o", type=str, default="dataset", help="Base dataset output directory")
    parser.add_argument("--num", "-n", type=int, default=150, help="Number of frames to sample")
    
    args = parser.parse_args()
    
    sample_frames(args.input, args.output, args.num)
