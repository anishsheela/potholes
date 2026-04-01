import os
import subprocess
import argparse
import time
import csv
import json
import re
from datetime import datetime

# List of models specified by the user to tune
MODELS_TO_TUNE = [
    # Modern CNNs (proven for fine-grained tasks)
    'convnext_tiny',
    'convnext_small',
    'efficientnetv2_s',
    
    # Your current best (keep for comparison)
    'resnet18',
    
    # Alternative approaches
    'swin_tiny_patch4_window7_224',
]

def main():
    parser = argparse.ArgumentParser(description='Tune Road Condition Classifiers')
    parser.add_argument('--data-dir', type=str, default='dataset/classification/training', help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=15, help='Epochs per model')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--merge-classes', action='store_true', help='Merge into Binary classification')
    parser.add_argument('--use-class-weights', action='store_true', help='Handle class imbalance')
    parser.add_argument('--val-split', type=float, default=0.3, help='Validation split ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    
    args = parser.parse_args()

    print(f"Starting Tuning Session with {len(MODELS_TO_TUNE)} Models:")
    print("=" * 50)
    for m in MODELS_TO_TUNE:
        print(f"- {m}")
    print("=" * 50)
    
    tuning_start_time = time.time()
    results = {}

    for i, model_name in enumerate(MODELS_TO_TUNE):
        print(f"\n[{i+1}/{len(MODELS_TO_TUNE)}] Tuning Model: {model_name}")
        print("-" * 50)
        
        # Build command for train_classifier.py
        cmd = [
            "python", "src/train_classifier.py",
            "--model", model_name,
            "--data-dir", args.data_dir,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
            "--val-split", str(args.val_split),
            "--seed", str(args.seed),
            "--patience", str(args.patience)
        ]
        
        if args.merge_classes:
            cmd.append("--merge-classes")
            
        if args.use_class_weights:
            cmd.append("--use-class-weights")
            
        print(f"Running command: {' '.join(cmd)}")
        
        # Run the training script
        try:
            start_train = time.time()
            
            # Use Popen to stream output line-by-line while collecting it
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, # Merge stderr into stdout
                text=True,
                bufsize=1, # Line buffered
                universal_newlines=True
            )
            
            full_output = []
            
            # Print output live to terminal 
            for line in process.stdout:
                print(line, end='', flush=True)
                full_output.append(line)
                
            process.wait()
            
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)
                
            time_taken = time.time() - start_train
            
            # Reconstruct the full output for parsing
            full_output_str = "".join(full_output)
            
            # Extract JSON metrics payload
            metrics_match = re.search(r'--- JSON_METRICS_START ---\n(.*?)\n--- JSON_METRICS_END ---', full_output_str, re.DOTALL)
            
            if metrics_match:
                metrics_data = json.loads(metrics_match.group(1))
                metrics_data['status'] = f"Success"
                results[model_name] = metrics_data
            else:
                results[model_name] = {
                    "status": "Success (No Metrics found)",
                    "time_seconds": time_taken
                }
                
            print(f"\n✅ Finished {model_name} in {time_taken//60:.0f}m {time_taken%60:.0f}s")
            
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Error training {model_name}. Skipping to next.")
            results[model_name] = {"status": "Failed"}
            
    # Tuning Summary
    total_time = time.time() - tuning_start_time
    print("\n" + "=" * 50)
    print(f"Tuning Session Complete in {total_time//60:.0f}m {total_time%60:.0f}s!")
    print("=" * 50)
    for model, data in results.items():
        if isinstance(data, dict) and data.get("status", "") == "Failed":
             print(f"{model.ljust(25)}: Failed")
        else:
             time_m, time_s = divmod(data.get('time_seconds', 0), 60)
             print(f"{model.ljust(25)}: {data.get('status', 'Success')} ({time_m:.0f}m {time_s:.0f}s)")
             
    print("\nTrained models are saved in the models/ directory.")

    # Save to CSV
    os.makedirs('processed_data', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"processed_data/tuning_results_{timestamp}.csv"
    
    csv_headers = [
        'Model', 'Status', 'Time (s)', 'Epochs', 'LR', 'Batch Size', 
        'Validation Acc (%)', 'Precision (Macro %)', 'Recall (Macro %)', 'F1 Score (Macro %)'
    ]
    
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(csv_headers)
        
        for model, data in results.items():
            if data.get('status') == 'Failed':
                writer.writerow([model, 'Failed'] + [''] * 8)
                continue
                
            writer.writerow([
                model,
                data.get('status', ''),
                f"{data.get('time_seconds', 0):.1f}",
                data.get('epochs', ''),
                data.get('lr', ''),
                data.get('batch_size', ''),
                f"{data.get('accuracy', 0):.2f}" if 'accuracy' in data else '',
                f"{data.get('precision', 0):.2f}" if 'precision' in data else '',
                f"{data.get('recall', 0):.2f}" if 'recall' in data else '',
                f"{data.get('f1_score', 0):.2f}" if 'f1_score' in data else ''
            ])
            
    print(f"\n✅ Tuning results saved to {csv_filename}")

if __name__ == '__main__':
    main()
