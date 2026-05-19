# -*- coding: utf-8 -*-
"""
collect_camera_samples.py - 采集真实摄像头 RGB 样本用于后续微调

示例：
    python collect_camera_samples.py --label genuine --camera_id 0 --interval 5 --max_images 300
    python collect_camera_samples.py --label attack --camera_id 0 --interval 5 --max_images 300
"""

import argparse
import csv
import os
import time
from datetime import datetime
from pathlib import Path

import cv2

from infer_onnx_camera import (
    open_camera,
    create_face_detector,
    detect_largest_face,
    crop_face_region,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / 'data' / 'real_camera_finetune'
LABEL_TO_ID = {
    'attack': 0,
    'genuine': 1,
}


def parse_args():
    parser = argparse.ArgumentParser(description='Collect real camera samples')
    parser.add_argument('--label', choices=['attack', 'genuine'], required=True)
    parser.add_argument('--camera_id', type=int, default=0)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--interval', type=int, default=5, help='Save every N frames')
    parser.add_argument('--max_images', type=int, default=300)
    parser.add_argument('--output_root', type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument('--face_crop', action='store_true')
    parser.add_argument('--crop_margin', type=float, default=0.25)
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    session = datetime.now().strftime('%Y%m%d_%H%M%S')
    image_dir = output_root / args.label / session
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / 'manifest.csv'

    capture = open_camera(args.camera_id, args.width, args.height)
    detector = create_face_detector() if args.face_crop else None
    saved = 0
    frame_index = 0

    print('[INFO] Press Q or ESC to stop.')
    print(f'[INFO] Saving to: {image_dir}')

    try:
        while saved < args.max_images:
            ok, frame = capture.read()
            if not ok or frame is None:
                print('[WARN] Failed to read frame.')
                break

            frame_index += 1
            display = frame.copy()
            save_frame = frame
            face_box_text = ''

            if detector is not None:
                detected_box = detect_largest_face(frame, detector)
                save_frame, crop_box = crop_face_region(frame, detected_box, args.crop_margin)
                if crop_box is not None:
                    x, y, w, h = [int(v) for v in crop_box]
                    face_box_text = f'{x},{y},{w},{h}'
                    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 215, 255), 2)

            if frame_index % max(1, args.interval) == 0:
                filename = f'{args.label}_{session}_{saved:04d}.jpg'
                image_path = image_dir / filename
                cv2.imwrite(str(image_path), save_frame)
                with manifest_path.open('a', encoding='utf-8', newline='') as f:
                    writer = csv.writer(f)
                    if manifest_path.stat().st_size == 0:
                        writer.writerow(['image_path', 'label', 'label_id', 'session', 'face_box'])
                    writer.writerow([
                        str(image_path.relative_to(output_root)).replace('\\', '/'),
                        args.label,
                        LABEL_TO_ID[args.label],
                        session,
                        face_box_text,
                    ])
                saved += 1

            cv2.putText(
                display,
                f'{args.label} saved {saved}/{args.max_images}',
                (18, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 215, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow('Collect Camera Samples', display)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break
            time.sleep(0.001)
    finally:
        capture.release()
        cv2.destroyAllWindows()

    print(f'[INFO] Saved images: {saved}')
    print(f'[INFO] Manifest: {manifest_path}')


if __name__ == '__main__':
    main()
