#!/usr/bin/env python3
import os
import sqlite3
import random
import torch
import timm
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import json
from collections import defaultdict
import torch.nn.functional as F

def build_file_map():
    search_dirs = [
        'processed_data/frames',
        'unfiltered_images',
        'road_classifier/unfiltered_images',
        'road_classifier/active_learning_images'
    ]
    file_map = {}
    for d in search_dirs:
        if os.path.exists(d):
            for root, _, files in os.walk(d):
                for f in files:
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        file_map[f] = os.path.join(root, f)
    return file_map

def get_consensus_data(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT image_name, username, label FROM classifications')
    rows = c.fetchall()
    conn.close()

    img_labels = defaultdict(list)
    for img, user, label in rows:
        img_labels[img].append(label)

    consensus_data = []
    for img, labels in img_labels.items():
        unique_labels = set(labels)
        for label in unique_labels:
            if labels.count(label) >= 2: # 2-vote consensus
                consensus_data.append((img, label))
                break
    return consensus_data

def generate_thesis_samples():
    db_path = 'road_classifier/classifications.db'
    weights_path = 'models/vit_small_patch16_224_epochs100_acc100.pth'
    output_dir = 'thesis_samples'
    
    if not os.path.exists(db_path):
        print(f"Error: DB not found at {db_path}")
        return
    if not os.path.exists(weights_path):
        print(f"Error: Weights not found at {weights_path}")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    CLASSES = ["Excellent", "Fair", "Good", "Invalid", "Poor"]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Loading ViT-Small model on {device}...")
    
    model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=len(CLASSES))
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
    except Exception as e:
        print(f"Failed to load weights: {e}")
        return
        
    model = model.to(device)
    model.eval()
    
    # Matching transforms
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    consensus_data = get_consensus_data(db_path)
    print(f"Loaded {len(consensus_data)} consensus ground truth annotations.")
    
    file_map = build_file_map()
    
    results = []
    print("Running inference...")
    with torch.no_grad():
        for img_rel_path, gt_label in consensus_data:
            basename = os.path.basename(img_rel_path)
            src_path = file_map.get(basename)
            if not src_path and os.path.exists(img_rel_path):
                src_path = img_rel_path
                
            if not src_path:
                continue
                
            try:
                img = Image.open(src_path).convert('RGB')
                tensor = data_transforms(img).unsqueeze(0).to(device)
                outputs = model(tensor)
                probs = F.softmax(outputs, dim=1)[0]
                conf, pred_idx = torch.max(probs, 0)
                pred_label = CLASSES[pred_idx.item()]
                results.append({
                    'path': src_path,
                    'gt': gt_label,
                    'pred': pred_label,
                    'conf': conf.item()
                })
            except Exception as e:
                pass
                
    print(f"Successfully processed {len(results)} images.")
    
    # Filter into the 3 buckets
    correct_high_conf = [r for r in results if r['pred'] == r['gt'] and r['conf'] >= 0.6]
    error_high_conf = [r for r in results if r['pred'] != r['gt'] and r['conf'] >= 0.6]
    low_conf = [r for r in results if r['conf'] < 0.5]
    
    # Shuffle so we get different samples each time
    random.seed(42) # Fixed seed to match user's report expectations or just for reproducible "junk"
    random.shuffle(correct_high_conf)
    random.shuffle(error_high_conf)
    random.shuffle(low_conf)
    
    # Select specific classes if possible for Row 1 (Excellent, Good, Invalid, Fair)
    row1 = []
    targets = ['Excellent', 'Good', 'Invalid', 'Fair', 'Poor']
    for tgt in targets:
        for r in correct_high_conf:
            if r['gt'] == tgt and r not in row1:
                row1.append(r)
                break
        if len(row1) == 4:
            break
    # Fill remaining if needed
    while len(row1) < 4 and correct_high_conf:
        r = correct_high_conf.pop(0)
        if r not in row1: row1.append(r)
        
    # Select errors for Row 2
    row2 = error_high_conf[:4]
    
    # Select low conf for Row 3
    row3 = low_conf[:4]
    
    grid = [row1, row2, row3]
    print(f"Selected Row 1: {len(row1)} items")
    print(f"Selected Row 2: {len(row2)} items")
    print(f"Selected Row 3: {len(row3)} items")
    
    # Create Composite Image Grid
    cell_w, cell_h = 300, 350
    margin = 10
    total_w = 4 * cell_w + 5 * margin
    total_h = 3 * cell_h + 4 * margin
    
    composite = Image.new('RGB', (total_w, total_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(composite)
    
    try:
        font = ImageFont.truetype("Arial", 16)
    except:
        font = ImageFont.load_default()
        
    for r_idx, row in enumerate(grid):
        for c_idx, item in enumerate(row):
            x = margin + c_idx * (cell_w + margin)
            y = margin + r_idx * (cell_h + margin)
            
            # Load and resize image safely keeping aspect ratio optionally or crop
            try:
                img = Image.open(item['path']).convert('RGB')
                # Resize to fit area 300x250
                img = img.resize((cell_w, 250))
                composite.paste(img, (x, y))
            except:
                continue
                
            # Draw Text
            status_char = "✓" if item['pred'] == item['gt'] else "✗"
            text_y = y + 260
            text_pred = f"Pred: {item['pred']} ({item['conf']:.2f})"
            text_gt = f"GT: {item['gt']} {status_char}"
            
            # Use red color for incorrect, green for correct
            color = (0, 150, 0) if item['pred'] == item['gt'] else (200, 0, 0)
            
            draw.text((x, text_y), text_pred, font=font, fill=(0,0,0))
            draw.text((x, text_y + 25), text_gt, font=font, fill=color)
            
            # Save individual copy
            flat_name = f"row{r_idx+1}_col{c_idx+1}_{item['pred']}_vs_{item['gt']}.jpg"
            img.save(os.path.join(output_dir, flat_name))
            
    out_img = os.path.join(output_dir, "Figure2_Grid.jpg")
    composite.save(out_img)
    print(f"\nSaved composite grid to {out_img}")
    print(f"Individual images saved in {output_dir}/")

    # Output detailed JSON mapping
    with open(os.path.join(output_dir, 'samples_metadata.json'), 'w') as f:
        json.dump({
            "correct_predictions_row1": [{"path": i["path"], "gt": i["gt"], "pred": i["pred"], "conf": i["conf"]} for i in row1],
            "high_confidence_errors_row2": [{"path": i["path"], "gt": i["gt"], "pred": i["pred"], "conf": i["conf"]} for i in row2],
            "low_confidence_predictions_row3": [{"path": i["path"], "gt": i["gt"], "pred": i["pred"], "conf": i["conf"]} for i in row3]
        }, f, indent=4)

if __name__ == '__main__':
    generate_thesis_samples()
