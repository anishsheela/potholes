"""
Phase 2: Hyperparameter tuning for a specific set of models.

Run this after model_sweep to narrow down which architectures are worth tuning.

Example:
    python src/classifier/hyperparam_tune.py \
        --models swin_small_patch4_window7_224,swin_tiny_patch4_window7_224 \
        --try-both-weights \
        --include-invalid \
        --lr-values 3e-4 1e-4 3e-5 \
        --freeze-epoch-values 0 5 \
        --scheduler cosine \
        --label-smoothing 0.1

The script sweeps every combination of (model, lr, batch_size, class_weights,
include_invalid, freeze_epochs) and saves results to
tune_classifier/hyperparam/<timestamp>/.
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

# Default search grid — override with CLI flags
LR_VALUES = [1e-3, 3e-4, 1e-4, 3e-5]
BATCH_SIZES = [16, 32, 64]
FREEZE_EPOCH_VALUES = [0, 5]


def run_trial(model_name, lr, batch_size, freeze_epochs,
              use_class_weights, include_invalid, args, log_dir):
    weight_tag   = 'weighted'     if use_class_weights else 'unweighted'
    invalid_tag  = 'with_invalid' if include_invalid   else 'no_invalid'
    freeze_tag   = f'freeze{freeze_epochs}'
    trial_id = (
        f'{model_name}__lr{lr}__bs{batch_size}'
        f'__{weight_tag}__{invalid_tag}__{freeze_tag}'
    )
    log_path = os.path.join(log_dir, f'{trial_id}.txt')

    cmd = [
        'python', TRAIN_SCRIPT,
        '--model',      model_name,
        '--data-dir',   args.data_dir,
        '--epochs',     str(args.epochs),
        '--batch-size', str(batch_size),
        '--lr',         str(lr),
        '--val-split',  str(args.val_split),
        '--seed',       str(args.seed),
        '--patience',   str(args.patience),
        '--label-smoothing', str(args.label_smoothing),
        '--freeze-epochs',   str(freeze_epochs),
        '--scheduler',       args.scheduler,
    ]
    if args.merge_classes:
        cmd.append('--merge-classes')
    if use_class_weights:
        cmd.append('--use-class-weights')
    if include_invalid:
        cmd.append('--include-invalid')

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
                'freeze_epochs': freeze_epochs,
                'class_weights': use_class_weights, 'include_invalid': include_invalid,
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
            metrics['lr'] = lr
            metrics['batch_size'] = batch_size
            metrics['freeze_epochs'] = freeze_epochs
            metrics['class_weights'] = use_class_weights
            metrics['include_invalid'] = include_invalid
            return metrics

        return {
            'model': model_name, 'lr': lr, 'batch_size': batch_size,
            'freeze_epochs': freeze_epochs,
            'class_weights': use_class_weights, 'include_invalid': include_invalid,
            'status': 'Success (no metrics)', 'time_seconds': elapsed, 'log': log_path,
        }

    except Exception as e:
        elapsed = time.time() - start
        return {
            'model': model_name, 'lr': lr, 'batch_size': batch_size,
            'freeze_epochs': freeze_epochs,
            'class_weights': use_class_weights, 'include_invalid': include_invalid,
            'status': f'Error: {e}', 'time_seconds': elapsed, 'log': log_path,
        }


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for selected models')
    parser.add_argument('--models', type=str, required=True,
                        help='Comma-separated list of model names, e.g. swin_small,swin_tiny')
    parser.add_argument('--data-dir', type=str, default='dataset/classification/training')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--merge-classes', action='store_true')
    parser.add_argument('--use-class-weights', action='store_true',
                        help='Use weighted loss. Ignored if --try-both-weights is set.')
    parser.add_argument('--include-invalid', action='store_true',
                        help='Include Invalid class. Ignored if --try-both-invalid is set.')
    parser.add_argument('--try-both-weights', action='store_true',
                        help='Run each trial twice: with and without class weights')
    parser.add_argument('--try-both-invalid', action='store_true',
                        help='Run each trial twice: with and without Invalid class')
    parser.add_argument('--val-split', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--lr-values', type=float, nargs='+', default=LR_VALUES,
                        help='Learning rates to sweep')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=BATCH_SIZES,
                        help='Batch sizes to sweep')
    parser.add_argument('--freeze-epoch-values', type=int, nargs='+', default=FREEZE_EPOCH_VALUES,
                        help='Freeze-epoch counts to sweep, e.g. --freeze-epoch-values 0 5 10')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                        help='Label smoothing for CrossEntropyLoss (0 = off)')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'plateau', 'none'],
                        help='LR scheduler for the main training phase')
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(',') if m.strip()]

    weight_variants = [True, False] if args.try_both_weights else [args.use_class_weights]
    invalid_variants = [True, False] if args.try_both_invalid else [args.include_invalid]

    combos = list(itertools.product(
        models,
        args.lr_values,
        args.batch_sizes,
        args.freeze_epoch_values,
        weight_variants,
        invalid_variants,
    ))
    total_trials = len(combos)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(OUTPUT_ROOT, timestamp)
    log_dir = os.path.join(run_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    config = {
        'models': models,
        'lr_values': args.lr_values,
        'batch_sizes': args.batch_sizes,
        'freeze_epoch_values': args.freeze_epoch_values,
        'weight_variants': weight_variants,
        'invalid_variants': invalid_variants,
        'try_both_weights': args.try_both_weights,
        'try_both_invalid': args.try_both_invalid,
        'label_smoothing': args.label_smoothing,
        'scheduler': args.scheduler,
        'epochs': args.epochs,
        'val_split': args.val_split,
        'seed': args.seed,
        'patience': args.patience,
        'merge_classes': args.merge_classes,
        'data_dir': args.data_dir,
        'timestamp': timestamp,
        'total_trials': total_trials,
    }
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print(f'\nHyperparameter Tuning — {total_trials} trials')
    print(f'  Models:        {models}')
    print(f'  LR values:     {args.lr_values}')
    print(f'  Batch sizes:   {args.batch_sizes}')
    print(f'  Freeze epochs: {args.freeze_epoch_values}')
    print(f'  Class weights: {weight_variants}')
    print(f'  Invalid class: {invalid_variants}')
    print(f'  Label smooth:  {args.label_smoothing}')
    print(f'  Scheduler:     {args.scheduler}')
    print(f'  Results dir:   {run_dir}')
    print('=' * 60)

    sweep_start = time.time()
    all_results = []

    for i, (model_name, lr, batch_size, freeze_epochs, use_weights, incl_invalid) in enumerate(combos):
        weight_label  = 'weighted'     if use_weights   else 'unweighted'
        invalid_label = 'with_invalid' if incl_invalid  else 'no_invalid'
        print(
            f'\n[{i+1}/{total_trials}] model={model_name}  lr={lr}  bs={batch_size}'
            f'  freeze={freeze_epochs}  weights={weight_label}  invalid={invalid_label}'
        )
        print('-' * 60)

        result = run_trial(
            model_name, lr, batch_size, freeze_epochs,
            use_weights, incl_invalid, args, log_dir
        )
        all_results.append(result)

        acc = result.get('accuracy', '')
        f1  = result.get('f1_score', '')
        t   = result.get('time_seconds', 0)
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

    # CSV — one row per trial
    csv_path = os.path.join(run_dir, 'summary.csv')
    csv_headers = [
        'model', 'lr', 'batch_size', 'freeze_epochs', 'class_weights', 'include_invalid',
        'status', 'epochs_trained',
        'val_accuracy_%', 'precision_macro_%', 'recall_macro_%', 'f1_macro_%',
        'time_seconds', 'log_file',
    ]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers, extrasaction='ignore')
        writer.writeheader()
        for d in all_results:
            writer.writerow({
                'model':            d.get('model', ''),
                'lr':               d.get('lr', ''),
                'batch_size':       d.get('batch_size', ''),
                'freeze_epochs':    d.get('freeze_epochs', ''),
                'class_weights':    d.get('class_weights', ''),
                'include_invalid':  d.get('include_invalid', ''),
                'status':           d.get('status', ''),
                'epochs_trained':   d.get('epochs', ''),
                'val_accuracy_%':   f"{d['accuracy']:.4f}"  if 'accuracy'  in d else '',
                'precision_macro_%':f"{d['precision']:.4f}" if 'precision' in d else '',
                'recall_macro_%':   f"{d['recall']:.4f}"    if 'recall'    in d else '',
                'f1_macro_%':       f"{d['f1_score']:.4f}"  if 'f1_score'  in d else '',
                'time_seconds':     f"{d.get('time_seconds', 0):.1f}",
                'log_file':         d.get('log', ''),
            })

    # Best-per-model table keyed by model + weight + invalid + freeze variant
    best_per_variant = {}
    for d in all_results:
        if 'accuracy' not in d:
            continue
        m  = d.get('model', '')
        w  = 'weighted'     if d.get('class_weights')   else 'unweighted'
        iv = 'w/invalid'    if d.get('include_invalid') else 'no_invalid'
        fz = f"freeze{d.get('freeze_epochs', '?')}"
        key = f'{m}__{w}__{iv}__{fz}'
        if key not in best_per_variant or d['f1_score'] > best_per_variant[key]['f1_score']:
            best_per_variant[key] = d

    print('\n' + '=' * 75)
    print(f'Tuning complete in {total_time//60:.0f}m {total_time%60:.0f}s')
    print('=' * 75)
    print(f'\nBest config per variant (by val F1):')
    print(f'{"Model":<35} {"Wts":<12} {"Inv":<12} {"Frz":>5} {"LR":>8} {"Acc%":>7} {"F1%":>7}')
    print('-' * 90)
    for key, d in sorted(best_per_variant.items(),
                         key=lambda x: x[1].get('f1_score', 0), reverse=True):
        w  = 'weighted'  if d.get('class_weights')   else 'unweighted'
        iv = 'w/invalid' if d.get('include_invalid') else 'no_invalid'
        fz = d.get('freeze_epochs', '?')
        print(
            f'{d["model"]:<35} {w:<12} {iv:<12} {fz:>5} {d["lr"]:>8} '
            f'{d["accuracy"]:>7.2f} {d.get("f1_score", 0):>7.2f}'
        )

    print(f'\nFull results: {run_dir}')
    print(f'  summary.json  — all trials + log paths')
    print(f'  summary.csv   — spreadsheet-ready, one row per trial')
    print(f'  logs/         — full training output per trial')


if __name__ == '__main__':
    main()
