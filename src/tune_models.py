import os
import subprocess
import csv

MODELS = [
    "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt",
    "yolov9t.pt", "yolov9s.pt", "yolov9m.pt", "yolov9c.pt",
    "yolov10n.pt", "yolov10s.pt", "yolov10m.pt", "yolov10l.pt",
    "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt",
    "yolov12n.pt", "yolov12s.pt", "yolov12m.pt", "yolov12l.pt",
    "yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt"
]

def parse_results(run_name, model_name):
    # Parse results.csv to get final metrics metrics from the run.
    # We dynamically look for the final epoch line to grab the results.
    results_path = os.path.join("runs", "segment", run_name, "results.csv")
    if not os.path.exists(results_path):
        print(f"Warning: No results found at {results_path}")
        return None
    
    with open(results_path, 'r') as f:
        reader = csv.reader(f)
        try:
            # First line is headers
            headers = [h.strip() for h in next(reader)]
            last_row = None
            for row in reader:
                if row: last_row = [v.strip() for v in row]
                
            if last_row:
                metrics = dict(zip(headers, last_row))
                metrics["model"] = model_name
                return metrics
        except Exception as e:
            print(f"Failed to parse results.csv: {e}")
            return None
    return None

def main():
    epochs = 70 # Modify this to match the number of epochs you want to tune for
    bg_ratio = "0.5"
    output_csv = "tuning_results.csv"
    
    headers_written = os.path.exists(output_csv)
        
    for index, model in enumerate(MODELS):
        print(f"\n{'='*50}")
        print(f"Training model {index+1}/{len(MODELS)}: {model}")
        print(f"{'='*50}\n")
        
        base_name = os.path.splitext(model)[0]
        run_name = f"tune_{base_name}"
        save_weights = f"models/best_{base_name}.pt"
        
        cmd = [
            "python", "src/train_yolo.py",
            "--model", model,
            "--epochs", str(epochs),
            "--run-name", run_name,
            "--bg-ratio", bg_ratio,
            "--save-weights", save_weights
        ]
        
        # After the first iteration, we reuse the exact dataset split for parity comparisons
        if index > 0:
            cmd.append("--reuse-split")
            
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        
        if result.returncode != 0:
            print(f"Warning: Training failed for model {model}. Moving to the next model.")
            continue
            
        metrics = parse_results(run_name, model)
        if metrics:
            print(f"Recorded metrics for {model}.")
            with open(output_csv, 'a', newline='') as f:
                # Add 'model' to beginning of the CSV headers
                fieldnames = ["model"] + [k for k in metrics.keys() if k != "model"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                if not headers_written:
                    writer.writeheader()
                    headers_written = True
                writer.writerow(metrics)
        else:
            print(f"Warning: Could not extract metrics for {model}.")
            
    print(f"\nPerformance tuning complete! Results saved to {output_csv}")

if __name__ == "__main__":
    main()
