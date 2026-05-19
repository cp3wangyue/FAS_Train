# -*- coding: utf-8 -*-
"""
summarize_gui_runs.py - 汇总 GUI 实时检测实验日志

读取 weights/gui_runs/run_*/session_log.csv，输出：
    1. weights/gui_runs/gui_runs_summary.csv
    2. weights/gui_runs/gui_runs_summary.md

该脚本不依赖 pandas，部署端也可以直接运行。
"""

import argparse
import csv
import os
from statistics import mean


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RUN_ROOT = os.path.join(PROJECT_ROOT, 'weights', 'gui_runs')


def parse_args():
    parser = argparse.ArgumentParser(description='Summarize FAS GUI run logs')
    parser.add_argument(
        '--run_root',
        type=str,
        default=DEFAULT_RUN_ROOT,
        help='Directory containing run_*/session_log.csv files',
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        default=None,
        help='Output CSV path, default: run_root/gui_runs_summary.csv',
    )
    parser.add_argument(
        '--output_md',
        type=str,
        default=None,
        help='Output Markdown path, default: run_root/gui_runs_summary.md',
    )
    parser.add_argument(
        '--min_records',
        type=int,
        default=10,
        help='Ignore runs with fewer records, default: 10',
    )
    return parser.parse_args()


def to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def collect_log_files(run_root):
    if not os.path.isdir(run_root):
        return []

    log_files = []
    for name in sorted(os.listdir(run_root)):
        run_dir = os.path.join(run_root, name)
        if not os.path.isdir(run_dir):
            continue
        csv_path = os.path.join(run_dir, 'session_log.csv')
        if os.path.exists(csv_path):
            log_files.append(csv_path)
    return log_files


def summarize_one(csv_path, min_records: int):
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))

    if len(rows) < min_records:
        return None

    display_fps = [to_float(r.get('display_fps')) for r in rows]
    infer_fps = [to_float(r.get('infer_fps')) for r in rows]
    inference_ms = [to_float(r.get('inference_ms')) for r in rows]
    brightness = [to_float(r.get('brightness_mean')) for r in rows]
    face_detected = [to_int(r.get('face_detected')) for r in rows]

    first = rows[0]
    run_name = os.path.basename(os.path.dirname(csv_path))
    face_detection_rate = mean(face_detected) if face_detected else 0.0

    return {
        'run_name': run_name,
        'records': len(rows),
        'start_time': first.get('timestamp', ''),
        'end_time': rows[-1].get('timestamp', ''),
        'capture_width': first.get('capture_width', ''),
        'capture_height': first.get('capture_height', ''),
        'infer_every_n_frames': first.get('infer_every_n_frames', ''),
        'low_light_enabled': first.get('low_light_enabled', ''),
        'face_crop_enabled': first.get('face_crop_enabled', ''),
        'face_detection_rate': f'{face_detection_rate:.4f}',
        'avg_display_fps': f'{mean(display_fps):.2f}',
        'avg_infer_fps': f'{mean(infer_fps):.2f}',
        'avg_inference_ms': f'{mean(inference_ms):.2f}',
        'avg_brightness': f'{mean(brightness):.2f}',
        'log_path': csv_path,
    }


def write_csv(summaries, output_csv):
    fieldnames = [
        'run_name',
        'records',
        'start_time',
        'end_time',
        'capture_width',
        'capture_height',
        'infer_every_n_frames',
        'low_light_enabled',
        'face_crop_enabled',
        'face_detection_rate',
        'avg_display_fps',
        'avg_infer_fps',
        'avg_inference_ms',
        'avg_brightness',
        'log_path',
    ]
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def write_markdown(summaries, output_md):
    os.makedirs(os.path.dirname(output_md), exist_ok=True)
    with open(output_md, 'w', encoding='utf-8') as f:
        f.write('# GUI 实时检测实验日志汇总\n\n')
        if not summaries:
            f.write('未找到可汇总的 `session_log.csv`。\n')
            return

        f.write('| 运行目录 | 记录数 | 分辨率 | 跳帧 | 低照度 | 人脸裁剪 | 人脸检出率 | Display FPS | Infer FPS | 推理耗时(ms) | 亮度 |\n')
        f.write('| --- | ---: | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |\n')
        for item in summaries:
            resolution = f"{item['capture_width']}x{item['capture_height']}"
            f.write(
                f"| {item['run_name']} | {item['records']} | {resolution} | "
                f"{item['infer_every_n_frames']} | {item['low_light_enabled']} | "
                f"{item['face_crop_enabled']} | {item['face_detection_rate']} | "
                f"{item['avg_display_fps']} | {item['avg_infer_fps']} | "
                f"{item['avg_inference_ms']} | {item['avg_brightness']} |\n"
            )

        f.write('\n说明：该表由 GUI 自动记录的 CSV 汇总生成，可作为论文第 5 章端侧测试表的原始依据。\n')


def main():
    args = parse_args()
    run_root = os.path.abspath(args.run_root)
    output_csv = args.output_csv or os.path.join(run_root, 'gui_runs_summary.csv')
    output_md = args.output_md or os.path.join(run_root, 'gui_runs_summary.md')

    summaries = []
    for csv_path in collect_log_files(run_root):
        summary = summarize_one(csv_path, args.min_records)
        if summary is not None:
            summaries.append(summary)

    write_csv(summaries, output_csv)
    write_markdown(summaries, output_md)

    print(f'[INFO] Found runs: {len(summaries)}')
    print(f'[INFO] Summary CSV: {output_csv}')
    print(f'[INFO] Summary MD:  {output_md}')


if __name__ == '__main__':
    main()
