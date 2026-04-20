"""
Phase 2: Hyperparameter tuning for a specific set of models.

Run this after model_sweep to narrow down which architectures are worth tuning.

Example:
    python src/classifier/hyperparam_tune.py \
        --models resnet50,convnext_tiny,efficientnetv2_s \
        --use-class-weights

The script sweeps every (lr, batch_size) combination for each model and saves
detailed results to tune_classifier/hyperparam/<timestamp>/.
"""

import os
import subprocess
import argparse
import time
import csv
import json
import re
import itertools
from datetime import datetime

TRAIN_SCRIPT = 'src/classifier/train_classifier.py'
OUTPUT_ROOT = 'tune_classifier/hyperparam'

# Search grid — edit to taste
LR_VALUES = [1e-3, 3e-4, 1e-4, 3e-5]
BATCH_SIZES = [16, 32, 64]


def run_trial(model_name, lr, batch_size, args, log_dir):
    trial_id = f'{model_name}__lr{lr}__bs{batch_size}'
    log_path = os.path.join(log_dir, f'{trial_id}.txt')

    cmd = [
        'python', TRAIN_SCRIPT,
        '--model', model_name,
        '--data-dir', args.data_dir,
        '--epochs', str(args.epochs),
        '--batch-size', str(batch_size),
        '--lr', str(lr),
        '--val-split', str(args.val_split),
        '--seed', str(args.seed),
        '--patience', str(args.patience),
    ]
    if args.merge_classes:
        cmd.append('--merge-classes')
    if args.use_class_weights:
        cmd.append('--use-class-weights')

    print(f'  Running: {" ".join(cmd)}')

    start = time.time()
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        lines = []
        for line in process.stdout:
            print(line, end='', flush=True)
            lines.append(line)

        process.wait()
        elapsed = time.time() - start
        full_output = ''.join(lines)

        with open(log_path, 'w') as f:
            f.write(f'Command: {" ".join(cmd)}\n')
            f.write(f'Exit code: {process.returncode}\n')
            f.write('=' * 60 + '\n')
            f.write(full_output)

        if process.returncode != 0:
            return {
                'model': model_name, 'lr': lr, 'batch_size': batch_size,
                'status': 'Failed', 'time_seconds': elapsed, 'log': log_path,
            }

        match = re.search(
            r'--- JSON_METRICS_START ---\n(.*?)\n--- JSON_METRICS_END ---',
            full_output,
            re.DOTALL,
        )
        if match:
            metrics = json.loads(match.group(1))
            metrics['status'] = 'Success'
            metrics['log'] = log_path
            # Overwrite lr/batch_size with the actual values used (not from JSON,
            # which should match anyway, but be explicit)
            metrics['lr'] = lr
            metrics['batch_size'] = batch_size
            return metrics

        return {
            'model': model_name, 'lr': lr, 'batch_size': batch_size,
            'status': 'Success (no metrics)', 'time_seconds': elapsed, 'log': log_path,
        }

    except Exception as e:
        elapsed = time.time() - start
        return {
            'model': model_name, 'lr': lr, 'batch_size': batch_size,
            'status': f'Error: {e}', 'time_seconds': elapsed, 'log': log_path,
        }


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for selected models')
    parser.add_argument('--models', type=str, required=True,
                        help='Comma-separated list of model names to tune, e.g. resnet50,convnext_tiny')
    parser.add_argument('--data-dir', type=str, default='dataset/classification/training')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--merge-classes', action='store_true')
    parser.add_argument('--use-class-weights', action='store_true')
    parser.add_argument('--val-split', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patience', type=int, default=3)
    # LR / batch-size are swept from the grid above; you can override the grid
    # by passing --lr-values and --batch-sizes as space-separated lists.
    parser.add_argument('--lr-values', type=float, nargs='+', default=LR_VALUES,
                        help='Learning rates to sweep')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=BATCH_SIZES,
                        help='Batch sizes to sweep')
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(',') if m.strip()]
    combos = list(itertools.product(models, args.lr_values, args.batch_sizes))
    total_trials = len(combos)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(OUTPUT_ROOT, timestamp)
    log_dir = os.path.join(run_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    config = {
        'models': models,
        'lr_values': args.lr_values,
        'batch_sizes': args.batch_sizes,
        'epochs': args.epochs,
        'val_split': args.val_split,
        'seed': args.seed,
        'patience': args.patience,
        'merge_classes': args.merge_classes,
        'use_class_weights': args.use_class_weights,
        'data_dir': args.data_dir,
        'timestamp': timestamp,
        'total_trials': total_trials,
    }
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print(f'\nHyperparameter Tuning — {total_trials} trials')
    print(f'  Models:      {models}')
    print(f'  LR values:   {args.lr_values}')
    print(f'  Batch sizes: {args.batch_sizes}')
    print(f'  Results dir: {run_dir}')
    print('=' * 60)

    sweep_start = time.time()
    all_results = []

    for i, (model_name, lr, batch_size) in enumerate(combos):
        print(f'\n[{i+1}/{total_trials}] model={model_name}  lr={lr}  batch_size={batch_size}')
        print('-' * 60)

        result = run_trial(model_name, lr, batch_size, args, log_dir)
        all_results.append(result)

        acc = result.get('accuracy', '')
        f1 = result.get('f1_score', '')
        t = result.get('time_seconds', 0)
        if isinstance(acc, float):
            print(f'\n  --> acc={acc:.2f}%  f1={f1:.2f}%  time={t//60:.0f}m{t%60:.0f}s')
        else:
            print(f'\n  --> {result.get("status")}')

        # Checkpoint after every trial
        with open(os.path.join(run_dir, 'results.json'), 'w') as f:
            json.dump(all_results, f, indent=2)

    total_time = time.time() - sweep_start

    # Final JSON
    summary = {
        'config': config,
        'total_time_seconds': total_time,
        'results': all_results,
    }
    with open(os.path.join(run_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # CSV
    csv_path = os.path.join(run_dir, 'summary.csv')
    csv_headers = [
        'model', 'lr', 'batch_size', 'status', 'epochs_trained',
        'val_accuracy_%', 'precision_macro_%', 'recall_macro_%', 'f1_macro_%',
        'time_seconds', 'log_file',
    ]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers, extrasaction='ignore')
        writer.writeheader()
        for d in all_results:
            writer.writerow({
                'model': d.get('model', ''),
                'lr': d.get('lr', ''),
                'batch_size': d.get('batch_size', ''),
                'status': d.get('status', ''),
                'epochs_trained': d.get('epochs', ''),
                'val_accuracy_%': f"{d['accuracy']:.4f}" if 'accuracy' in d else '',
                'precision_macro_%': f"{d['precision']:.4f}" if 'precision' in d else '',
                'recall_macro_%': f"{d['recall']:.4f}" if 'recall' in d else '',
                'f1_macro_%': f"{d['f1_score']:.4f}" if 'f1_score' in d else '',
                'time_seconds': f"{d.get('time_seconds', 0):.1f}",
                'log_file': d.get('log', ''),
            })

    # Best-per-model table
    best_per_model = {}
    for d in all_results:
        m = d.get('model', '')
        if 'accuracy' not in d:
            continue
        if m not in best_per_model or d['accuracy'] > best_per_model[m]['accuracy']:
            best_per_model[m] = d

    print('\n' + '=' * 60)
    print(f'Tuning complete in {total_time//60:.0f}m {total_time%60:.0f}s')
    print('=' * 60)
    print(f'\nBest config per model (by val accuracy):')
    print(f'{"Model":<35} {"LR":>8} {"BS":>4} {"Acc%":>7} {"F1%":>7}')
    print('-' * 65)
    for model_name, d in sorted(best_per_model.items(), key=lambda x: x[1].get('accuracy', 0), reverse=True):
        print(
            f'{model_name:<35} {d["lr"]:>8} {d["batch_size"]:>4} '
            f'{d["accuracy"]:>7.2f} {d.get("f1_score", 0):>7.2f}'
        )

    print(f'\nFull results: {run_dir}')
    print(f'  summary.json  — all trials + log paths')
    print(f'  summary.csv   — spreadsheet-ready, one row per trial')
    print(f'  logs/         — full training output per trial')


if __name__ == '__main__':
    main()
