# AI Dashcam Pothole Detection & Route Mapping

An end-to-end Python pipeline that processes uncalibrated dashcam videos, extracts text telemetry via Optical Character Recognition (OCR), maps the driven route with GPS coordinates, and trains a custom YOLOv8 Vision AI to automatically detect potholes along the mapped roads.

## Pipeline Overview

This project is divided into two continuous phases:
1. **Data Extraction & Mapping:** Raw `.MP4` dashcam videos are processed. High-speed OCR (`pytesseract`) reads the burned-in GPS coordinates, speed, and timestamp from every single frame. Idle/Nighttime frames and GPS anomalies are discarded, and the true driven route is mapped using `folium`.
2. **Vision Model Training:** A custom dataset is randomly sampled from the extracted frames. Using MakeSense.ai for manual validation and `ultralytics` for model fine-tuning, a custom **YOLOv8 Nano** object detection model is trained. The final model processes raw video and draws bounding boxes around potholes in real-time.

---

## 🛠️ Prerequisites & Installation

You will need **Python 3.10+** and **FFmpeg** installed on your system.
*On macOS:* `brew install ffmpeg tesseract`

1. Clone the repository
2. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install the required Python packages (OpenCV, Pytesseract, Pandas, Folium, Ultralytics, Geopy):
   ```bash
   pip install opencv-python pytesseract pandas folium ultralytics geopy tqdm
   ```

---

## 📂 Phase 1: Video Extraction & GPS Mapping

Place your raw dashcam videos in the `videos/` folder. The pipeline scripts are fully *incremental*, meaning you can add new videos later and immediately re-run the pipeline without re-processing old data!

**1. Extract Frames:**  
Extracts 1 frame per second from all videos using hardware-accelerated FFmpeg.
```bash
python src/extract_frames.py -i videos -o output/frames
```

**2. OCR Processor (Multi-Core):**  
Reads the text telemetry from the bottom corners of every extracted frame and compiles it into `output/route_data.csv`.
```bash
python src/ocr_processor.py -i output/frames -o output/route_data.csv
```

**3. Data Filtering:**  
Cleans the dataset by removing idle frames (0 km/h), nighttime frames (before 6 AM / after 6 PM), and impossible GPS jumps based on Geodesic distance. 
```bash
python src/filter_frames_by_speed.py \
    --csv output/route_data.csv \
    --frames-dir output/frames \
    --out-csv output/filtered_route_data.csv \
    --out-dir output/filtered_frames
```

**4. Generate Route Map:**  
Plots the cleaned GPS points onto an interactive `folium` web map.
```bash
python src/generate_map.py -i output/filtered_route_data.csv -o output/map.html
```

---

## 🧠 Phase 2: Custom YOLOv8 Pothole Detection

Once Phase 1 is complete, you will have a massive folder (`output/filtered_frames/`) of clean, daylight, forward-facing road images.

**1. Sample an Annotation Batch:**  
Extract 150 random images into a new "to-do" folder (`dataset/unannotated/batch_X/images/`). The script keeps a history, so you will *never* sample the same frame twice.
```bash
python src/sample_frames.py -i output/filtered_frames -o dataset -n 150
```

**2. Manual Annotation:**
1. Open [MakeSense.ai](https://www.makesense.ai/) in your browser.
2. Drag and drop the 150 frames from your new batch folder.
3. Draw tight bounding boxes around all visible potholes. *(If a frame has no potholes, leave it completely blank! This serves as a vital negative background sample).*
4. Export the annotations in YOLO format (`.txt` files).
5. Place the exported files into a new `labels/` directory next to your `images/` directory.
6. Move the completed batch folder from `dataset/unannotated/` into `dataset/annotated/`.

**3. Train the Model!**  
The training script will seamlessly crawl your `annotated/` directory, combine all your batches, generate the validation split, and fine-tune a pre-trained YOLOv8 Nano model. It automatically utilizes PyTorch hardware acceleration (`MPS` for Mac or `CUDA` for Nvidia) if available.
```bash
python src/train_yolo.py --data-dir dataset --epochs 50
```

**4. Run Video Inference:**  
Test your custom AI on a raw `.MP4` file. It will draw boxes around detected potholes and spit out a new, annotated video.
```bash
python src/predict_video.py -v videos/YOUR_VIDEO.MP4
```

Alternatively, you can test the model directly against a raw directory of images:
```bash
python src/predict_frames.py -i output/filtered_frames/YOUR_VIDEO_FOLDER
```
