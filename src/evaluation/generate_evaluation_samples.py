#!/usr/bin/env python3
"""
Generate random sample of predictions for manual evaluation.
Creates a simple HTML interface for quick yes/no labeling.
"""

import os
import glob
import random
import shutil
import argparse
from pathlib import Path

def create_evaluation_set(annotated_dir, output_dir, num_samples=200):
    """
    Creates a random sample of annotated images for manual evaluation.
    
    Args:
        annotated_dir: Directory containing annotated prediction images
        output_dir: Where to save the evaluation sample
        num_samples: Number of random images to sample
    """
    # Find all annotated images
    all_images = []
    for ext in ['*.jpg', '*.png']:
        all_images.extend(glob.glob(os.path.join(annotated_dir, '**', ext), recursive=True))
    
    if not all_images:
        print(f"No images found in {annotated_dir}")
        return
    
    print(f"Found {len(all_images)} total annotated images")
    
    # Random sample
    sample_size = min(num_samples, len(all_images))
    sampled_images = random.sample(all_images, sample_size)
    
    # Create output directory
    eval_images_dir = os.path.join(output_dir, 'images')
    os.makedirs(eval_images_dir, exist_ok=True)
    
    # Copy sampled images
    print(f"Copying {sample_size} random images...")
    image_list = []
    for i, img_path in enumerate(sampled_images):
        basename = os.path.basename(img_path)
        # Add index to avoid name collisions
        new_name = f"{i:04d}_{basename}"
        dest_path = os.path.join(eval_images_dir, new_name)
        shutil.copy2(img_path, dest_path)
        image_list.append(new_name)
    
    # Create HTML evaluation interface
    create_html_interface(output_dir, image_list)
    
    # Create CSV template for results
    create_csv_template(output_dir, image_list)
    
    print(f"\n✅ Evaluation set created at: {output_dir}")
    print(f"   Images: {eval_images_dir}")
    print(f"   Open: {os.path.join(output_dir, 'evaluate.html')}")

def create_html_interface(output_dir, image_list):
    """Creates an HTML interface for quick evaluation."""
    
    html_content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Pothole Detection Evaluation</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .image-container {
            text-align: center;
            margin: 20px 0;
            background: #000;
            padding: 20px;
            border-radius: 5px;
        }
        .image-container img {
            max-width: 100%;
            max-height: 600px;
            border: 2px solid #ddd;
        }
        .controls {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin: 20px 0;
            flex-wrap: wrap;
        }
        button {
            padding: 15px 30px;
            font-size: 16px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: all 0.3s;
            font-weight: bold;
        }
        .btn-tp { background: #4CAF50; color: white; }
        .btn-tn { background: #2196F3; color: white; }
        .btn-fp { background: #FF9800; color: white; }
        .btn-fn { background: #f44336; color: white; }
        button:hover { opacity: 0.8; transform: scale(1.05); }
        .progress {
            margin: 20px 0;
            font-size: 18px;
            font-weight: bold;
            color: #333;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .stat-card {
            background: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
        }
        .stat-card h3 { margin: 0 0 10px 0; color: #666; font-size: 14px; }
        .stat-card .value { font-size: 24px; font-weight: bold; color: #333; }
        .export-btn {
            background: #9C27B0;
            color: white;
            padding: 12px 25px;
            margin: 20px 0;
        }
        .help {
            background: #e3f2fd;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
            border-left: 4px solid #2196F3;
        }
        .help h3 { margin-top: 0; }
        .help ul { margin: 10px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🕳️ Pothole Detection Evaluation</h1>
        
        <div class="help">
            <h3>How to Evaluate:</h3>
            <ul>
                <li><strong>True Positive (TP):</strong> Model correctly detected pothole(s)</li>
                <li><strong>True Negative (TN):</strong> No potholes in image, model correctly predicted none</li>
                <li><strong>False Positive (FP):</strong> Model detected pothole(s), but there are none (wrong!)</li>
                <li><strong>False Negative (FN):</strong> Potholes exist, but model missed them (wrong!)</li>
            </ul>
            <p><strong>Note:</strong> We're evaluating image-level detection (Does frame have pothole? Yes/No), not box accuracy.</p>
        </div>
        
        <div class="progress">
            Image <span id="current">1</span> of <span id="total">""" + str(len(image_list)) + """</span>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <h3>True Positives</h3>
                <div class="value" id="tp-count">0</div>
            </div>
            <div class="stat-card">
                <h3>True Negatives</h3>
                <div class="value" id="tn-count">0</div>
            </div>
            <div class="stat-card">
                <h3>False Positives</h3>
                <div class="value" id="fp-count">0</div>
            </div>
            <div class="stat-card">
                <h3>False Negatives</h3>
                <div class="value" id="fn-count">0</div>
            </div>
        </div>
        
        <div class="image-container">
            <img id="current-image" src="" alt="Loading...">
            <p id="image-name" style="color: white; margin-top: 10px;"></p>
        </div>
        
        <div class="controls">
            <button class="btn-tp" onclick="classify('TP')">✓ True Positive (TP)</button>
            <button class="btn-tn" onclick="classify('TN')">✓ True Negative (TN)</button>
            <button class="btn-fp" onclick="classify('FP')">✗ False Positive (FP)</button>
            <button class="btn-fn" onclick="classify('FN')">✗ False Negative (FN)</button>
        </div>
        
        <div style="text-align: center;">
            <button class="export-btn" onclick="exportResults()">💾 Export Results (CSV)</button>
        </div>
    </div>

    <script>
        const images = """ + str(image_list) + """;
        let currentIndex = 0;
        let results = [];
        let stats = { TP: 0, TN: 0, FP: 0, FN: 0 };

        // Load results from localStorage if available
        if (localStorage.getItem('evaluation_results')) {
            const saved = JSON.parse(localStorage.getItem('evaluation_results'));
            results = saved.results;
            stats = saved.stats;
            currentIndex = saved.currentIndex;
            updateStats();
        }

        function loadImage() {
            if (currentIndex >= images.length) {
                alert('Evaluation complete! Click Export to download results.');
                return;
            }
            
            const img = images[currentIndex];
            document.getElementById('current-image').src = 'images/' + img;
            document.getElementById('image-name').textContent = img;
            document.getElementById('current').textContent = currentIndex + 1;
        }

        function classify(label) {
            results.push({
                image: images[currentIndex],
                label: label,
                timestamp: new Date().toISOString()
            });
            
            stats[label]++;
            updateStats();
            saveProgress();
            
            currentIndex++;
            loadImage();
        }

        function updateStats() {
            document.getElementById('tp-count').textContent = stats.TP;
            document.getElementById('tn-count').textContent = stats.TN;
            document.getElementById('fp-count').textContent = stats.FP;
            document.getElementById('fn-count').textContent = stats.FN;
        }

        function saveProgress() {
            localStorage.setItem('evaluation_results', JSON.stringify({
                results: results,
                stats: stats,
                currentIndex: currentIndex
            }));
        }

        function exportResults() {
            // Calculate metrics
            const total = stats.TP + stats.TN + stats.FP + stats.FN;
            const accuracy = total > 0 ? ((stats.TP + stats.TN) / total * 100).toFixed(2) : 0;
            const precision = (stats.TP + stats.FP) > 0 ? (stats.TP / (stats.TP + stats.FP) * 100).toFixed(2) : 0;
            const recall = (stats.TP + stats.FN) > 0 ? (stats.TP / (stats.TP + stats.FN) * 100).toFixed(2) : 0;
            const f1 = (parseFloat(precision) + parseFloat(recall)) > 0 ? 
                       (2 * parseFloat(precision) * parseFloat(recall) / (parseFloat(precision) + parseFloat(recall))).toFixed(2) : 0;
            
            // Create CSV content
            let csv = 'Image,Label,Timestamp\\n';
            results.forEach(r => {
                csv += `${r.image},${r.label},${r.timestamp}\\n`;
            });
            
            csv += '\\n# Summary Statistics\\n';
            csv += `Total Evaluated,${total}\\n`;
            csv += `True Positives,${stats.TP}\\n`;
            csv += `True Negatives,${stats.TN}\\n`;
            csv += `False Positives,${stats.FP}\\n`;
            csv += `False Negatives,${stats.FN}\\n`;
            csv += '\\n# Metrics\\n';
            csv += `Accuracy,${accuracy}%\\n`;
            csv += `Precision,${precision}%\\n`;
            csv += `Recall,${recall}%\\n`;
            csv += `F1 Score,${f1}%\\n`;
            
            // Download CSV
            const blob = new Blob([csv], { type: 'text/csv' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'evaluation_results_' + new Date().toISOString().split('T')[0] + '.csv';
            a.click();
            
            alert(`Results exported!\\n\\nAccuracy: ${accuracy}%\\nPrecision: ${precision}%\\nRecall: ${recall}%\\nF1 Score: ${f1}%`);
        }

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === '1') classify('TP');
            if (e.key === '2') classify('TN');
            if (e.key === '3') classify('FP');
            if (e.key === '4') classify('FN');
        });

        // Load first image
        loadImage();
    </script>
</body>
</html>"""
    
    with open(os.path.join(output_dir, 'evaluate.html'), 'w') as f:
        f.write(html_content)

def create_csv_template(output_dir, image_list):
    """Creates a CSV template for manual evaluation."""
    csv_path = os.path.join(output_dir, 'evaluation_template.csv')
    
    with open(csv_path, 'w') as f:
        f.write('image,label,notes\n')
        for img in image_list:
            f.write(f'{img},,\n')
    
    print(f"   CSV template: {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate evaluation sample from predictions")
    parser.add_argument("--input", "-i", required=True, 
                       help="Directory containing annotated predictions")
    parser.add_argument("--output", "-o", default="evaluation_sample",
                       help="Output directory for evaluation set (default: evaluation_sample)")
    parser.add_argument("--num-samples", "-n", type=int, default=200,
                       help="Number of random samples (default: 200)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    create_evaluation_set(args.input, args.output, args.num_samples)