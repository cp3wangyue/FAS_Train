# -*- coding: utf-8 -*-
"""
infer_onnx_batch.py - ONNX 批量推理脚本

功能：
    1. 支持从 CASIA-SURF 官方 list 文件批量推理
    2. 支持从图片目录递归扫描批量推理
    3. 输出逐样本 CSV 结果
    4. 若输入含真实标签，则自动统计 Accuracy / APCER / BPCER / ACER
    5. 可选保存每张图片的伪深度图

示例：
    conda run -n FAS_Train python D:/FAS_Train/infer_onnx_batch.py ^
        --list_file D:/FAS_Train/data/CASIA-SURF/train/train_list.txt ^
        --limit 20 --save_depth
"""

import os
import csv
import json
import time
import argparse
from typing import Optional

import numpy as np

from infer_onnx import (
    PROJECT_ROOT,
    LABEL_NAMES,
    preprocess_image,
    build_session,
    run_inference,
    softmax,
    save_depth_map,
)


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def parse_args():
    parser = argparse.ArgumentParser(description='Batch inference with ONNX')
    parser.add_argument(
        '--onnx_path',
        type=str,
        default=os.path.join(PROJECT_ROOT, 'weights', 'best_model.onnx'),
        help='Path to ONNX model, default: weights/best_model.onnx'
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--list_file',
        type=str,
        default=None,
        help='CASIA-SURF style txt list file, or plain image-path list'
    )
    input_group.add_argument(
        '--image_dir',
        type=str,
        default=None,
        help='Directory containing images for recursive inference'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Threshold on genuine probability, default: 0.5'
    )
    parser.add_argument(
        '--base_dir',
        type=str,
        default=None,
        help='Optional base dir for resolving relative paths in list_file'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Optional max number of samples to process'
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        default=os.path.join(PROJECT_ROOT, 'weights', 'infer_outputs', 'batch_results.csv'),
        help='CSV path for batch inference results'
    )
    parser.add_argument(
        '--summary_json',
        type=str,
        default=None,
        help='Optional JSON path for aggregate metrics summary'
    )
    parser.add_argument(
        '--save_depth',
        action='store_true',
        help='Whether to save predicted depth maps'
    )
    parser.add_argument(
        '--depth_dir',
        type=str,
        default=os.path.join(PROJECT_ROOT, 'weights', 'infer_outputs', 'batch_depth'),
        help='Output directory for depth maps'
    )
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='When using --image_dir, recursively scan subdirectories'
    )
    return parser.parse_args()


def is_binary_label(text: str) -> bool:
    return text in {'0', '1'}


def collect_from_list(list_file: str, base_dir: Optional[str] = None) -> list[dict]:
    list_file = os.path.abspath(list_file)
    list_root = os.path.abspath(base_dir) if base_dir else os.path.dirname(list_file)
    samples = []

    with open(list_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            true_label: Optional[int] = None
            rel_path = parts[0]

            # CASIA-SURF 官方格式：rgb depth ir label
            if len(parts) >= 4 and is_binary_label(parts[3]):
                rel_path = parts[0]
                true_label = int(parts[3])
            # 兼容：image_path label
            elif len(parts) >= 2 and is_binary_label(parts[-1]):
                rel_path = parts[0]
                true_label = int(parts[-1])

            image_path = rel_path
            if not os.path.isabs(image_path):
                image_path = os.path.abspath(os.path.join(list_root, image_path))

            samples.append({
                'image_path': image_path,
                'true_label': true_label,
                'rel_path': rel_path.replace('\\', '/'),
            })

    return samples


def collect_from_dir(image_dir: str, recursive: bool) -> list[dict]:
    image_dir = os.path.abspath(image_dir)
    samples = []

    if recursive:
        walker = os.walk(image_dir)
        for root, _, files in walker:
            for name in sorted(files):
                ext = os.path.splitext(name)[1].lower()
                if ext not in IMAGE_EXTENSIONS:
                    continue
                image_path = os.path.join(root, name)
                rel_path = os.path.relpath(image_path, image_dir).replace('\\', '/')
                samples.append({
                    'image_path': image_path,
                    'true_label': None,
                    'rel_path': rel_path,
                })
    else:
        for name in sorted(os.listdir(image_dir)):
            image_path = os.path.join(image_dir, name)
            if not os.path.isfile(image_path):
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext not in IMAGE_EXTENSIONS:
                continue
            samples.append({
                'image_path': image_path,
                'true_label': None,
                'rel_path': name,
            })

    return samples


def compute_pad_metrics(preds: list[int], labels: list[int]) -> dict:
    preds_np = np.array(preds)
    labels_np = np.array(labels)

    tp = int(((preds_np == 1) & (labels_np == 1)).sum())
    tn = int(((preds_np == 0) & (labels_np == 0)).sum())
    fp = int(((preds_np == 1) & (labels_np == 0)).sum())
    fn = int(((preds_np == 0) & (labels_np == 1)).sum())

    n_attack = int((labels_np == 0).sum())
    n_genuine = int((labels_np == 1).sum())
    total = len(labels)

    accuracy = (tp + tn) / total if total > 0 else 0.0
    apcer = fp / n_attack if n_attack > 0 else 0.0
    bpcer = fn / n_genuine if n_genuine > 0 else 0.0
    acer = (apcer + bpcer) / 2.0

    return {
        'accuracy': accuracy,
        'apcer': apcer,
        'bpcer': bpcer,
        'acer': acer,
        'confusion_matrix': {
            'TP': tp,
            'TN': tn,
            'FP': fp,
            'FN': fn,
            'N_attack': n_attack,
            'N_genuine': n_genuine,
        }
    }


def resolve_summary_json(output_csv: str, summary_json: Optional[str]) -> str:
    if summary_json:
        return os.path.abspath(summary_json)
    stem, _ = os.path.splitext(os.path.abspath(output_csv))
    return f'{stem}_summary.json'


def resolve_depth_path(depth_dir: str, rel_path: str) -> str:
    rel_norm = rel_path.replace('\\', '/')
    parts = [p for p in rel_norm.split('/') if p]
    rel_dirs = parts[:-1]
    stem = os.path.splitext(parts[-1])[0]
    filename = f'{stem}_depth.png'
    if rel_dirs:
        return os.path.join(depth_dir, *rel_dirs, filename)
    return os.path.join(depth_dir, filename)


def save_results_csv(records: list[dict], output_csv: str):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    fieldnames = [
        'index',
        'image_path',
        'true_label',
        'pred_label',
        'pred_name',
        'attack_prob',
        'genuine_prob',
        'inference_ms',
        'depth_min',
        'depth_max',
        'depth_mean',
        'depth_output',
        'is_correct',
    ]
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main():
    args = parse_args()

    onnx_path = os.path.abspath(args.onnx_path)
    output_csv = os.path.abspath(args.output_csv)
    summary_json = resolve_summary_json(output_csv, args.summary_json)
    depth_dir = os.path.abspath(args.depth_dir)

    if args.list_file:
        samples = collect_from_list(args.list_file, args.base_dir)
        input_desc = os.path.abspath(args.list_file)
    else:
        samples = collect_from_dir(args.image_dir, args.recursive)
        input_desc = os.path.abspath(args.image_dir)

    if args.limit is not None:
        samples = samples[:args.limit]

    if not samples:
        raise RuntimeError('No samples found for batch inference.')

    print('=' * 70)
    print('  FAS ONNX Batch Inference')
    print('=' * 70)
    print(f'  ONNX model:   {onnx_path}')
    print(f'  Input source: {input_desc}')
    print(f'  Samples:      {len(samples)}')
    print(f'  Threshold:    {args.threshold:.2f}')
    print(f'  Save depth:   {args.save_depth}')
    print(f'  Output CSV:   {output_csv}')
    print('=' * 70)

    session = build_session(onnx_path)

    records = []
    preds = []
    labels = []
    for idx, sample in enumerate(samples, start=1):
        input_tensor, _ = preprocess_image(sample['image_path'])
        t0 = time.perf_counter()
        cls_logits, depth_pred = run_inference(session, input_tensor)
        inference_ms = (time.perf_counter() - t0) * 1000.0

        probs = softmax(cls_logits)[0]
        attack_prob = float(probs[0])
        genuine_prob = float(probs[1])
        pred_label = 1 if genuine_prob >= args.threshold else 0
        pred_name = LABEL_NAMES[pred_label]

        depth_map = depth_pred[0, 0]
        depth_min = float(np.min(depth_map))
        depth_max = float(np.max(depth_map))
        depth_mean = float(np.mean(depth_map))

        depth_output = ''
        if args.save_depth:
            depth_output = resolve_depth_path(depth_dir, sample['rel_path'])
            save_depth_map(depth_map, depth_output)

        true_label = sample['true_label']
        is_correct = ''
        if true_label is not None:
            preds.append(pred_label)
            labels.append(true_label)
            is_correct = int(pred_label == true_label)

        records.append({
            'index': idx,
            'image_path': sample['image_path'],
            'true_label': '' if true_label is None else true_label,
            'pred_label': pred_label,
            'pred_name': pred_name,
            'attack_prob': f'{attack_prob:.6f}',
            'genuine_prob': f'{genuine_prob:.6f}',
            'inference_ms': f'{inference_ms:.2f}',
            'depth_min': f'{depth_min:.6f}',
            'depth_max': f'{depth_max:.6f}',
            'depth_mean': f'{depth_mean:.6f}',
            'depth_output': depth_output,
            'is_correct': is_correct,
        })

        if idx == 1 or idx % 50 == 0 or idx == len(samples):
            print(f'[Progress] {idx}/{len(samples)} done')

    save_results_csv(records, output_csv)
    print(f'[Save] Batch results CSV: {output_csv}')

    summary = {
        'onnx_path': onnx_path,
        'input_source': input_desc,
        'sample_count': len(samples),
        'threshold': args.threshold,
        'save_depth': args.save_depth,
        'output_csv': output_csv,
    }

    if labels:
        metrics = compute_pad_metrics(preds, labels)
        summary.update({
            'labeled_sample_count': len(labels),
            'accuracy': metrics['accuracy'],
            'apcer': metrics['apcer'],
            'bpcer': metrics['bpcer'],
            'acer': metrics['acer'],
            'confusion_matrix': metrics['confusion_matrix'],
        })
        print(f"[Summary] Accuracy={metrics['accuracy'] * 100:.2f}%")
        print(f"[Summary] APCER={metrics['apcer']:.4f}  "
              f"BPCER={metrics['bpcer']:.4f}  ACER={metrics['acer']:.4f}")
        cm = metrics['confusion_matrix']
        print(f"[Summary] CM: TP={cm['TP']} TN={cm['TN']} "
              f"FP={cm['FP']} FN={cm['FN']}")
    else:
        print('[Summary] No ground-truth labels found, skip PAD metrics.')

    os.makedirs(os.path.dirname(summary_json), exist_ok=True)
    with open(summary_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'[Save] Summary JSON: {summary_json}')


if __name__ == '__main__':
    main()
