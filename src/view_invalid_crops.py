import argparse
import pandas as pd
import cv2
import yaml
import os

def main():
    parser = argparse.ArgumentParser(description="View crops from invalid OCR frames.")
    parser.add_argument("--camera", type=str, required=True, help="Camera name (e.g., aravind)")
    parser.add_argument("--config", "-c", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    # Load config
    try:
        with open(args.config, 'r') as f:
            master_config = yaml.safe_load(f)
            camera_config = master_config['cameras'].get(args.camera)
            if not camera_config:
                print(f"Error: Camera '{args.camera}' not found in {args.config}")
                return
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    # Load coordinates
    gps_coords = camera_config['gps_roi']['coords']
    date_coords = camera_config['date_roi']['coords']
    
    # Load invalid data
    invalid_csv = os.path.join("processed_data", "route_data", f"invalid_gps_{args.camera}.csv")
    if not os.path.exists(invalid_csv):
        print(f"Error: Invalid CSV file not found: {invalid_csv}")
        return
        
    try:
        df = pd.read_csv(invalid_csv)
    except Exception as e:
        print(f"Error reading {invalid_csv}: {e}")
        return
        
    if 'Frame_Path' not in df.columns:
        print(f"Error: 'Frame_Path' column not found in {invalid_csv}. Did you run the updated OCR processor?")
        return

    print(f"Found {len(df)} invalid frames to review.")
    print("\nControls:")
    print("  'd' or 'Right Arrow' : Next image")
    print("  'a' or 'Left Arrow'  : Previous image")
    print("  'q' or 'ESC'         : Quit")
    
    cv2.namedWindow('OCR Debug Viewer', cv2.WINDOW_NORMAL)
    # Give the user a reasonably large window to interact with
    cv2.resizeWindow('OCR Debug Viewer', 800, 400)
    
    idx = 0
    while 0 <= idx < len(df):
        row = df.iloc[idx]
        frame_path = row['Frame_Path']
        raw_ocr = row['Raw_OCR']
        
        if not pd.isna(frame_path) and os.path.exists(frame_path):
            img = cv2.imread(frame_path)
            
            if img is not None:
                # Extract ROIs
                y1, y2, x1, x2 = gps_coords
                gps_crop = img[y1:y2, x1:x2]
                
                dy1, dy2, dx1, dx2 = date_coords
                date_crop = img[dy1:dy2, dx1:dx2]

                # Create a composite image to display both crops
                # Pad heights to match
                max_h = max(gps_crop.shape[0], date_crop.shape[0])
                
                # Create empty canvases to hold centered crops (for neatness)
                padded_gps = cv2.copyMakeBorder(gps_crop, 0, max_h - gps_crop.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                padded_date = cv2.copyMakeBorder(date_crop, 0, max_h - date_crop.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                
                # Add some spacing between crops
                spacer_width = 20
                spacer = cv2.copyMakeBorder(padded_date[:, 0:0], 0, 0, 0, spacer_width, cv2.BORDER_CONSTANT, value=[50, 50, 50])
                
                composite = cv2.hconcat([padded_date, spacer, padded_gps])
                
                # Add text info on top or bottom
                display_img = cv2.copyMakeBorder(composite, 60, 40, 20, 20, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                
                # Annotate
                info_text1 = f"Image {idx+1}/{len(df)}: {os.path.basename(frame_path)}"
                info_text2 = f"Raw OCR Output: '{raw_ocr}'"
                
                cv2.putText(display_img, info_text1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(display_img, info_text2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                
                # Scale up by 2x or 3x for visibility
                scale_factor = 3
                display_img = cv2.resize(display_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
                
                cv2.imshow('OCR Debug Viewer', display_img)
            else:
                print(f"Could not load image: {frame_path}")
                # Create a blank image with error message
                blank = cv2.copyMakeBorder(cv2.Mat.zeros(100, 400, cv2.CV_8UC3), 0,0,0,0, cv2.BORDER_CONSTANT, [0,0,0])
                cv2.putText(blank, f"Load Error: {os.path.basename(frame_path)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.imshow('OCR Debug Viewer', blank)
        else:
             print(f"File not found: {frame_path}")
             
        # Wait for key press
        key = cv2.waitKey(0)
        
        # Parse key
        if key == ord('d') or key == 83: # 'd' or Right Arrow
            idx += 1
        elif key == ord('a') or key == 81: # 'a' or Left Arrow
            idx -= 1
        elif key == ord('q') or key == 27: # 'q' or Escape
            break
            
        # Bounds checking loop around
        if idx >= len(df):
            print("Reached end of list.")
            idx -= 1
        if idx < 0:
            print("Already at the first image.")
            idx = 0

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
