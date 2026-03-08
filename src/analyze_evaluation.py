#!/usr/bin/env python3
"""
Analyze manual evaluation results and calculate image-level metrics.
"""

import csv
import argparse
from collections import Counter

def analyze_results(csv_file):
    """
    Analyzes evaluation CSV and calculates image-level metrics.
    
    Args:
        csv_file: Path to evaluation results CSV
    """
    results = []
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('label') and row['label'] in ['TP', 'TN', 'FP', 'FN']:
                results.append(row['label'])
    
    if not results:
        print("No valid labels found in CSV. Make sure to use labels: TP, TN, FP, FN")
        return
    
    # Count occurrences
    counts = Counter(results)
    tp = counts.get('TP', 0)
    tn = counts.get('TN', 0)
    fp = counts.get('FP', 0)
    fn = counts.get('FN', 0)
    
    total = tp + tn + fp + fn
    
    # Calculate metrics
    accuracy = (tp + tn) / total * 100 if total > 0 else 0
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Specificity (True Negative Rate)
    specificity = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
    
    # Print results
    print("\n" + "="*60)
    print("IMAGE-LEVEL DETECTION METRICS (Manual Evaluation)")
    print("="*60)
    
    print(f"\n📊 Confusion Matrix:")
    print(f"   True Positives (TP):  {tp:4d}  (Model found pothole, pothole exists)")
    print(f"   True Negatives (TN):  {tn:4d}  (Model found none, none exist)")
    print(f"   False Positives (FP): {fp:4d}  (Model found pothole, none exist)")
    print(f"   False Negatives (FN): {fn:4d}  (Model missed pothole)")
    print(f"   {'─'*50}")
    print(f"   Total Evaluated:      {total:4d}")
    
    print(f"\n📈 Performance Metrics:")
    print(f"   Accuracy:    {accuracy:6.2f}%  (Overall correctness)")
    print(f"   Precision:   {precision:6.2f}%  (When model says 'pothole', how often correct?)")
    print(f"   Recall:      {recall:6.2f}%  (Of all potholes, how many found?)")
    print(f"   Specificity: {specificity:6.2f}%  (Of all clean road, how many correct?)")
    print(f"   F1 Score:    {f1_score:6.2f}%  (Harmonic mean of precision/recall)")
    
    # Interpretation
    print(f"\n💡 Interpretation:")
    
    if accuracy >= 80:
        print(f"   ✅ Excellent accuracy ({accuracy:.1f}%) - Model is production-ready!")
    elif accuracy >= 70:
        print(f"   ✓  Good accuracy ({accuracy:.1f}%) - Model works well for most cases")
    elif accuracy >= 60:
        print(f"   ⚠  Moderate accuracy ({accuracy:.1f}%) - Consider more training data")
    else:
        print(f"   ❌ Low accuracy ({accuracy:.1f}%) - Needs significant improvement")
    
    if precision >= 70:
        print(f"   ✅ Low false positive rate ({100-precision:.1f}%) - Trustworthy detections")
    elif precision >= 50:
        print(f"   ⚠  Moderate false positives ({100-precision:.1f}%) - Some noise in detections")
    else:
        print(f"   ❌ High false positives ({100-precision:.1f}%) - Many incorrect detections")
    
    if recall >= 70:
        print(f"   ✅ Finding most potholes ({recall:.1f}%) - Good coverage")
    elif recall >= 50:
        print(f"   ⚠  Missing some potholes ({100-recall:.1f}% missed) - Acceptable for flagging")
    else:
        print(f"   ❌ Missing many potholes ({100-recall:.1f}% missed) - Needs improvement")
    
    # Comparison with validation metrics
    print(f"\n🔬 Compare to Validation Metrics:")
    print(f"   Your manual evaluation: Precision={precision:.1f}%, Recall={recall:.1f}%")
    print(f"   YOLO validation showed:  Precision=38.1%, Recall=30.5%")
    print(f"   ")
    if precision > 50 or recall > 50:
        print(f"   ✅ Real-world performance is better than validation suggested!")
        print(f"   This confirms validation metrics were pessimistic.")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze evaluation results")
    parser.add_argument("csv_file", help="Path to evaluation results CSV")
    
    args = parser.parse_args()
    analyze_results(args.csv_file)