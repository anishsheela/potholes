import cv2
import easyocr
import argparse
import os

def find_roi(image_path, output_path):
    print(f"Loading image {image_path}...")
    img = cv2.imread(image_path)
    if img is None:
        print("Error loading image.")
        return

    height, width, _ = img.shape
    print(f"Image dimensions: {width}x{height}")

    print("Initializing EasyOCR...")
    reader = easyocr.Reader(['en'], gpu=False) # Use CPU for now
    
    # Text is likely in the bottom 20% of the image
    bottom_crop_y = int(height * 0.8)
    cropped_img = img[bottom_crop_y:height, 0:width]
    
    print("Running OCR on the bottom 20% of the image...")
    results = reader.readtext(cropped_img)

    for (bbox, text, prob) in results:
        # Unpack bounding box
        (tl, tr, br, bl) = bbox
        tl = (int(tl[0]), int(tl[1]) + bottom_crop_y)
        br = (int(br[0]), int(br[1]) + bottom_crop_y)
        
        print(f"Detected: '{text}' (prob: {prob:.4f}) at bbox {tl} to {br}")
        
        # Draw bounding box on original image
        cv2.rectangle(img, tl, br, (0, 255, 0), 2)
        cv2.putText(img, text, (tl[0], tl[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imwrite(output_path, img)
    print(f"Saved annotated image to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to a sample frame")
    parser.add_argument("--output", type=str, default="output/debug/annotated_frame.jpg")
    args = parser.parse_args()
    
    find_roi(args.image, args.output)
