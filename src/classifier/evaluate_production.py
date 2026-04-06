#!/usr/bin/env python3
import os
import sqlite3
import json
import argparse
from collections import defaultdict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Model against Manual Web Verification')
    parser.add_argument('--predictions', type=str, default='eval_predictions.json', help='JSON output from predict_production.py mapping image -> Model Class')
    parser.add_argument('--db', type=str, default='road_classifier/classifications.db', help='SQLite database from road_classifier app.py')
    return parser.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(args.predictions):
        print(f"Error: {args.predictions} not found.")
        print("Please run predict_production.py first.")
        return

    if not os.path.exists(args.db):
        print(f"Error: Database {args.db} not found.")
        print("Please verify the images in the web UI first before running evaluation.")
        return

    # 1. Load Model Predictions
    with open(args.predictions, 'r') as f:
        model_preds = json.load(f)

    if not model_preds:
        print("Model predictions file is empty.")
        return

    print(f"Loaded model predictions for {len(model_preds)} images.")

    # 2. Load Human Annotations from DB
    conn = sqlite3.connect(args.db)
    c = conn.cursor()
    
    # We query the classifications table.
    # Group by image_name and take the most recent label in case it was annotated multiple times
    query = """
        SELECT image_name, label
        FROM classifications
        ORDER BY timestamp DESC
    """
    
    c.execute(query)
    rows = c.fetchall()
    conn.close()

    # Drop duplicates to keep only the newest label per image
    human_labels = {}
    for image_name, label in rows:
        if image_name not in human_labels:
            human_labels[image_name] = label
            
    print(f"Loaded human annotations for {len(human_labels)} images from DB.")

    # 3. Join the datasets
    y_true = []
    y_pred = []
    disagreements = []
    
    evaluated_count = 0
    missing_count = 0

    for image_name, model_class in model_preds.items():
        if image_name in human_labels:
            human_class = human_labels[image_name]
            y_true.append(human_class)
            y_pred.append(model_class)
            evaluated_count += 1
            
            if human_class != model_class:
                disagreements.append({
                    "image": image_name,
                    "human": human_class,
                    "model": model_class
                })
        else:
            missing_count += 1

    if evaluated_count == 0:
        print("\nNo overlapping images found between model predictions and database!")
        print("Are you sure you verified the images from 'eval_images' in the web interface yet?")
        return

    print(f"\nEvaluating on {evaluated_count} mutually available images (Missing: {missing_count}).")

    # 4. Generate Statistics
    CLASSES = ["Excellent", "Fair", "Good", "Invalid", "Poor"]
    
    acc = accuracy_score(y_true, y_pred)
    print("\n" + "="*50)
    print(f" OVERALL ACCURACY : {acc:.4f} ({acc*100:.1f}%)")
    print("="*50 + "\n")

    print("--- Classification Report ---")
    print(classification_report(y_true, y_pred, labels=CLASSES, zero_division=0))

    print("\n--- Confusion Matrix (True \\ Pred) ---")
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    
    # Print custom formatted confusion matrix since pandas is not available
    header = f"{'True / Pred':<15} | " + " | ".join([f"{c:<9}" for c in CLASSES])
    print(header)
    print("-" * len(header))
    
    for i, c_true in enumerate(CLASSES):
        row_str = f"{c_true:<15} | "
        row_vals = [f"{cm[i][j]:<9}" for j in range(len(CLASSES))]
        row_str += " | ".join(row_vals)
        print(row_str)

    # 5. Show Disagreements
    print("\n" + "="*50)
    print(f" DISAGREEMENTS ({len(disagreements)} total):")
    print("="*50)
    
    if len(disagreements) == 0:
        print("Perfect match! The model agreed with you 100% of the time.")
    else:
        for idx, d in enumerate(disagreements[:20]):
            print(f"  [{idx+1}] {d['image']}: You said '{d['human']}', Model said '{d['model']}'")
        
        if len(disagreements) > 20:
            print(f"  ... and {len(disagreements) - 20} more disagreements.")

if __name__ == '__main__':
    main()
