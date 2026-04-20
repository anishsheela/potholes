import os
import subprocess
import argparse
import time
import csv
import json
import re
from datetime import datetime

# All architectures supported by train_classifier.py
MODELS_TO_TUNE = [
    # ResNet family
    'resnet18',
    'resnet34',
    'resnet50',

    # EfficientNet family
    'efficientnet_b0',
    'efficientnet_b1',
    'efficientnet_b2',
    'efficientnetv2_s',
    'efficientnetv2_m',

    # Vision Transformers
    'vit_small_patch16_224',
    'vit_base_patch16_224',

    # ConvNeXt family
    'convnext_tiny',
    'convnext_small',
    'convnext_base',

    # Swin Transformers
    'swin_tiny_patch4_window7_224',
    'swin_small_patch4_window7_224',
]

TRAIN_SCRIPT = 'src/classifier/train_classifier.py'
OUTPUT_ROOT = 'tune_classifier/model_sweep'


def run_model(model_name, args, log_dir):
    cmd = [
        'python', TRAIN_SCRIPT,
        '--model', model_name,
        '--data-dir', args.data_dir,
        '--epochs', str(args.epochs),
        '--batch-size', str(args.batch_size),
        '--lr', str(args.lr),
        '--val-split', str(args.val_split),
        '--seed', str(args.seed),
        '--patience', str(args.patience),
    ]
    if args.merge_classes:
        cmd.append('--merge-classes')
    if args.use_class_weights:
        cmd.append('--use-class-weights')
    if args.include_invalid:
        cmd.append('--include-invalid')
    cmd += ['--label-smoothing', str(args.label_smoothing)]
    cmd += ['--freeze-epochs', str(args.freeze_epochs)]
    cmd += ['--scheduler', args.scheduler]

    log_path = os.path.join(log_dir, f'{model_name}.txt')

    print(f'Running: {" ".join(cmd)}')

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

        # Persist full log
        with open(log_path, 'w') as f:
            f.write(f'Command: {" ".join(cmd)}\n')
            f.write(f'Exit code: {process.returncode}\n')
            f.write('=' * 60 + '\n')
            f.write(full_output)

        if process.returncode != 0:
            return {'status': 'Failed', 'time_seconds': elapsed, 'log': log_path}

        # Parse JSON metrics emitted by train_classifier.py
        match = re.search(
            r'--- JSON_METRICS_START ---\n(.*?)\n--- JSON_METRICS_END ---',
            full_output,
            re.DOTALL,
        )
        if match:
            metrics = json.loads(match.group(1))
            metrics['status'] = 'Success'
            metrics['log'] = log_path
            return metrics

        return {'status': 'Success (no metrics)', 'time_seconds': elapsed, 'log': log_path}

    except Exception as e:
        elapsed = time.time() - start
        return {'status': f'Error: {e}', 'time_seconds': elapsed, 'log': log_path}


def main():
    parser = argparse.ArgumentParser(description='Sweep all model architectures for road condition classifier')
    parser.add_argument('--data-dir', type=str, default='dataset/classification/training')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--merge-classes', action='store_true')
    parser.add_argument('--use-class-weights', action='store_true')
    parser.add_argument('--include-invalid', action='store_true',
                        help='Include Invalid class in training')
    parser.add_argument('--val-split', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                        help='Label smoothing for CrossEntropyLoss (0 = off)')
    parser.add_argument('--freeze-epochs', type=int, default=5,
                        help='Epochs to train head only before unfreezing backbone (0 = off)')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'plateau', 'none'],
                        help='LR scheduler for main training phase')
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(OUTPUT_ROOT, timestamp)
    log_dir = os.path.join(run_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # Save run config
    config = vars(args)
    config['models'] = MODELS_TO_TUNE
    config['timestamp'] = timestamp
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print(f'\nModel Sweep — {len(MODELS_TO_TUNE)} models')
    print(f'Results will be saved to: {run_dir}')
    print('=' * 60)
    for m in MODELS_TO_TUNE:
        print(f'  {m}')
    print('=' * 60)

    sweep_start = time.time()
    results = {}

    for i, model_name in enumerate(MODELS_TO_TUNE):
        print(f'\n[{i+1}/{len(MODELS_TO_TUNE)}] {model_name}')
        print('-' * 60)

        result = run_model(model_name, args, log_dir)
        results[model_name] = result

        status = result.get('status', '')
        acc = result.get('accuracy', '')
        f1 = result.get('f1_score', '')
        t = result.get('time_seconds', 0)
        print(f'\n  --> {model_name}: status={status} | acc={acc:.2f}% | f1={f1:.2f}% | time={t//60:.0f}m{t%60:.0f}s'
              if isinstance(acc, float) else f'\n  --> {model_name}: {status}')

        # Checkpoint after each model so partial results are never lost
        with open(os.path.join(run_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)

    total_time = time.time() - sweep_start

    # Final JSON
    summary = {
        'config': config,
        'total_time_seconds': total_time,
        'results': results,
    }
    with open(os.path.join(run_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # CSV — one row per model, all metrics as columns
    csv_path = os.path.join(run_dir, 'summary.csv')
    csv_headers = [
        'model', 'status', 'epochs_trained', 'lr', 'batch_size',
        'val_accuracy_%', 'precision_macro_%', 'recall_macro_%', 'f1_macro_%',
        'time_seconds', 'log_file',
    ]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers, extrasaction='ignore')
        writer.writeheader()
        for model_name, d in results.items():
            writer.writerow({
                'model': model_name,
                'status': d.get('status', ''),
                'epochs_trained': d.get('epochs', ''),
                'lr': d.get('lr', args.lr),
                'batch_size': d.get('batch_size', args.batch_size),
                'val_accuracy_%': f"{d['accuracy']:.4f}" if 'accuracy' in d else '',
                'precision_macro_%': f"{d['precision']:.4f}" if 'precision' in d else '',
                'recall_macro_%': f"{d['recall']:.4f}" if 'recall' in d else '',
                'f1_macro_%': f"{d['f1_score']:.4f}" if 'f1_score' in d else '',
                'time_seconds': f"{d.get('time_seconds', 0):.1f}",
                'log_file': d.get('log', ''),
            })

    # Console summary table
    print('\n' + '=' * 60)
    print(f'Sweep complete in {total_time//60:.0f}m {total_time%60:.0f}s')
    print('=' * 60)
    print(f'{"Model":<35} {"Acc%":>7} {"F1%":>7} {"Status"}')
    print('-' * 60)
    for model_name, d in sorted(results.items(), key=lambda x: x[1].get('accuracy', 0), reverse=True):
        acc = f"{d['accuracy']:.2f}" if 'accuracy' in d else 'N/A'
        f1 = f"{d['f1_score']:.2f}" if 'f1_score' in d else 'N/A'
        print(f'{model_name:<35} {acc:>7} {f1:>7}   {d.get("status", "")}')

    print(f'\nFull results: {run_dir}')
    print(f'  summary.json  — all metrics + logs paths')
    print(f'  summary.csv   — spreadsheet-ready')
    print(f'  logs/         — full training output per model')


if __name__ == '__main__':
    main()
