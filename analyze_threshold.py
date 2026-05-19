# -*- coding: utf-8 -*-
"""
analyze_threshold.py - 根据 GUI 带标签日志分析最佳 Genuine 阈值

GUI 的 session_log.csv 中 true_label 为空时不会参与统计。
标签约定：
    0 = Attack
    1 = Genuine
"""

import argparse
import csv
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RUN_ROOT = PROJECT_ROOT / 'weights' / 'gui_runs'


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze threshold from labeled GUI logs')
    parser.add_argument('--run_root', type=str, default=str(DEFAULT_RUN_ROOT))
    parser.add_argument('--output_csv', type=str, default=None)
    parser.add_argument('--step', type=float, default=0.01)
    return parser.parse_args()


def iter_log_files(run_root: Path):
    for csv_path in sorted(run_root.glob('run_*/session_log.csv')):
        yield csv_path


def load_samples(run_root: Path):
    samples = []
    for csv_path in iter_log_files(run_root):
        with csv_path.open('r', encoding='utf-8', newline='') as f:
            for row in csv.DictReader(f):
                true_label = row.get('true_label', '').strip()
                if true_label not in {'0', '1'}:
                    continue
                try:
                    genuine_prob = float(row.get('genuine_prob', ''))
                except ValueError:
                    continue
                samples.append({
                    'run_name': csv_path.parent.name,
                    'true_label': int(true_label),
                    'genuine_prob': genuine_prob,
                })
    return samples


def compute_metrics(samples, threshold):
    tp = tn = fp = fn = 0
    for sample in samples:
        pred = 1 if sample['genuine_prob'] >= threshold else 0
        true = sample['true_label']
        if true == 1 and pred == 1:
            tp += 1
        elif true == 1 and pred == 0:
            fn += 1
        elif true == 0 and pred == 0:
            tn += 1
        elif true == 0 and pred == 1:
            fp += 1

    total = tp + tn + fp + fn
    n_attack = tn + fp
    n_genuine = tp + fn
    acc = (tp + tn) / total if total else 0.0
    apcer = fp / n_attack if n_attack else 0.0
    bpcer = fn / n_genuine if n_genuine else 0.0
    acer = (apcer + bpcer) / 2.0
    return {
        'threshold': threshold,
        'total': total,
        'attack_count': n_attack,
        'genuine_count': n_genuine,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'accuracy': acc,
        'apcer': apcer,
        'bpcer': bpcer,
        'acer': acer,
    }


def threshold_values(step):
    current = 0.0
    while current <= 1.000001:
        yield round(current, 4)
        current += step


def write_results(results, output_csv):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    fieldnames = [
        'threshold', 'total', 'attack_count', 'genuine_count',
        'tp', 'tn', 'fp', 'fn', 'accuracy', 'apcer', 'bpcer', 'acer',
    ]
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            formatted = dict(row)
            for key in ['accuracy', 'apcer', 'bpcer', 'acer']:
                formatted[key] = f"{row[key]:.6f}"
            writer.writerow(formatted)


def main():
    args = parse_args()
    run_root = Path(args.run_root).resolve()
    output_csv = args.output_csv or str(run_root / 'threshold_analysis.csv')
    samples = load_samples(run_root)

    if not samples:
        print('[WARN] No labeled samples found.')
        print('[HINT] In GUI, set Sample Label to Genuine or Attack before recording.')
        return

    results = [compute_metrics(samples, t) for t in threshold_values(args.step)]
    best = min(results, key=lambda r: (r['acer'], -r['accuracy'], r['threshold']))
    write_results(results, output_csv)

    print(f"[INFO] Labeled samples: {len(samples)}")
    print(f"[INFO] Attack: {best['attack_count']} | Genuine: {best['genuine_count']}")
    print(f"[BEST] threshold={best['threshold']:.2f} "
          f"acc={best['accuracy']:.4f} apcer={best['apcer']:.4f} "
          f"bpcer={best['bpcer']:.4f} acer={best['acer']:.4f}")
    print(f"[INFO] Saved: {output_csv}")


if __name__ == '__main__':
    main()
