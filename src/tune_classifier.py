import os
import subprocess
import argparse
import time

# List of models specified by the user to tune
MODELS_TO_TUNE = [
    'efficientnet_b0',
    'efficientnet_b1',
    'efficientnet_b2',
    'resnet18',
    'resnet34',
    'vit_base_patch16_224',
    'vit_small_patch16_224'
]

def main():
    parser = argparse.ArgumentParser(description='Tune Road Condition Classifiers')
    parser.add_argument('--data-dir', type=str, default='dataset/classification/training', help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=15, help='Epochs per model')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--merge-classes', action='store_true', help='Merge into Binary classification')
    parser.add_argument('--use-class-weights', action='store_true', help='Handle class imbalance')
    
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
            "--lr", str(args.lr)
        ]
        
        if args.merge_classes:
            cmd.append("--merge-classes")
            
        if args.use_class_weights:
            cmd.append("--use-class-weights")
            
        print(f"Running command: {' '.join(cmd)}")
        
        # Run the training script
        try:
            start_train = time.time()
            # Capture output so we can parse results if needed, or just pipe it to terminal
            process = subprocess.run(
                cmd,
                check=True,
                text=True,
                # Comment out stdout/stderr to let it print directly to the console during tuning
                # stdout=subprocess.PIPE,
                # stderr=subprocess.PIPE
            )
            time_taken = time.time() - start_train
            results[model_name] = f"Success ({time_taken//60:.0f}m {time_taken%60:.0f}s)"
            print(f"\n✅ Finished {model_name} in {time_taken//60:.0f}m {time_taken%60:.0f}s")
            
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Error training {model_name}. Skipping to next.")
            results[model_name] = "Failed"
            
    # Tuning Summary
    total_time = time.time() - tuning_start_time
    print("\n" + "=" * 50)
    print(f"Tuning Session Complete in {total_time//60:.0f}m {total_time%60:.0f}s!")
    print("=" * 50)
    for model, status in results.items():
        print(f"{model.ljust(25)}: {status}")
    print("\nTrained models are saved in the models/ directory.")

if __name__ == '__main__':
    main()
