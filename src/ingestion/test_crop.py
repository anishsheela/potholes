import cv2
import argparse
import os

def test_crop(image_path, top_crop_pct, bottom_crop_pct, output_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return
        
    h, w = img.shape[:2]
    
    # Calculate pixel boundaries based on exact percentages
    top_y = int(h * (top_crop_pct / 100.0))
    bottom_y = int(h * (1.0 - (bottom_crop_pct / 100.0)))
    
    # Slice the numpy array
    cropped = img[top_y:bottom_y, 0:w]
    
    cv2.imwrite(output_path, cropped)
    print(f"--- Crop Test Complete ---")
    print(f"Original Resolution: {w}x{h} pixels")
    print(f"Cropped Resolution:  {w}x{bottom_y - top_y} pixels")
    print(f"Saved visualization to: {output_path}")
    print(f"Open this file to verify the dashboard and sky are completely removed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ROI Crop limits on a dashcam frame")
    parser.add_argument("-i", "--image", required=True, help="Path to a sample dashcam frame .jpg")
    parser.add_argument("-t", "--top", type=float, default=30.0, help="Percentage to crop from TOP (0-100, default 30)")
    parser.add_argument("-b", "--bottom", type=float, default=35.0, help="Percentage to crop from BOTTOM (0-100, default 35)")
    parser.add_argument("-o", "--output", default="crop_test.jpg", help="Output visualization path")
    
    args = parser.parse_args()
    test_crop(args.image, args.top, args.bottom, args.output)
