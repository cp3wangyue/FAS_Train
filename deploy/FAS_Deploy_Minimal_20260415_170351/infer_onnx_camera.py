# -*- coding: utf-8 -*-
"""
infer_onnx_camera.py - ONNX 摄像头实时推理脚本

功能：
    1. 打开本地摄像头，持续读取视频帧
    2. 对每一帧执行与训练一致的 RGB 预处理
    3. 使用 ONNX Runtime 做真假分类 + 伪深度预测
    4. 在窗口上实时显示预测类别、概率、推理耗时和 FPS
    5. 可选显示单独的伪深度图窗口

使用示例：
    conda run -n FAS_Train python D:/FAS_Train/infer_onnx_camera.py
    conda run -n FAS_Train python D:/FAS_Train/infer_onnx_camera.py --camera_id 1 --show_depth
"""

import os
import time
import argparse

import cv2
import numpy as np

from infer_onnx import (
    PROJECT_ROOT,
    LABEL_NAMES,
    RGB_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_session,
    run_inference,
    softmax,
)


WINDOW_NAME = 'FAS Camera Inference'
DEPTH_WINDOW_NAME = 'FAS Depth Map'
PROB_SMOOTH_ALPHA = 0.35
DECISION_HYSTERESIS = 0.08
SWITCH_CONFIRM_FRAMES = 3
DEFAULT_ONNX_PATH = (
    os.path.join(PROJECT_ROOT, 'weights', 'best_model_me.onnx')
    if os.path.exists(os.path.join(PROJECT_ROOT, 'weights', 'best_model_me.onnx'))
    else os.path.join(PROJECT_ROOT, 'weights', 'best_model.onnx')
)


def parse_args():
    parser = argparse.ArgumentParser(description='Real-time camera inference with ONNX')
    parser.add_argument(
        '--onnx_path',
        type=str,
        default=DEFAULT_ONNX_PATH,
        help='Path to ONNX model, default: weights/best_model_me.onnx if available, otherwise weights/best_model.onnx'
    )
    parser.add_argument(
        '--camera_id',
        type=int,
        default=0,
        help='Camera index, default: 0'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.60,
        help='Threshold on genuine probability, default: 0.60'
    )
    parser.add_argument(
        '--width',
        type=int,
        default=1280,
        help='Requested capture width, default: 1280'
    )
    parser.add_argument(
        '--height',
        type=int,
        default=720,
        help='Requested capture height, default: 720'
    )
    parser.add_argument(
        '--flip',
        action='store_true',
        help='Horizontally flip the camera frame before inference and display'
    )
    parser.add_argument(
        '--show_depth',
        action='store_true',
        help='Show predicted depth map in a separate window'
    )
    parser.add_argument(
        '--face_crop',
        action='store_true',
        help='Detect and crop the largest face before ONNX inference'
    )
    parser.add_argument(
        '--allow_no_face_inference',
        action='store_true',
        help='When --face_crop is enabled, still run inference on the full frame if no face is detected.'
    )
    parser.add_argument(
        '--crop_margin',
        type=float,
        default=0.80,
        help='Extra margin ratio around detected face, default: 0.80'
    )
    return parser.parse_args()


def open_camera(camera_id: int, width: int, height: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    if not capture.isOpened():
        capture.release()
        capture = cv2.VideoCapture(camera_id)

    if not capture.isOpened():
        raise RuntimeError(
            f'Failed to open camera_id={camera_id}. '
            'Please check whether the camera is occupied, disconnected, or try --camera_id 1.'
        )

    try:
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    except Exception:
        pass

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, 30)
    if hasattr(cv2, 'CAP_PROP_BUFFERSIZE'):
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def preprocess_bgr_frame(frame_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (RGB_SIZE, RGB_SIZE), interpolation=cv2.INTER_LINEAR)
    rgb = rgb.astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    chw = np.transpose(rgb, (2, 0, 1))
    return np.expand_dims(chw, axis=0).astype(np.float32)


def create_face_detector() -> cv2.CascadeClassifier:
    cascade_path = os.path.join(
        cv2.data.haarcascades,
        'haarcascade_frontalface_default.xml',
    )
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError(f'Failed to load OpenCV face cascade: {cascade_path}')
    return detector


def detect_faces(frame_bgr: np.ndarray, detector: cv2.CascadeClassifier):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
    )
    return sorted(faces, key=lambda box: int(box[2]) * int(box[3]), reverse=True)


def detect_largest_face(frame_bgr: np.ndarray, detector: cv2.CascadeClassifier):
    faces = detect_faces(frame_bgr, detector)
    if not faces:
        return None
    return faces[0]


def crop_face_region(frame_bgr: np.ndarray, face_box, margin_ratio: float = 0.25):
    if face_box is None:
        return frame_bgr, None

    x, y, w, h = [int(v) for v in face_box]
    margin = int(max(w, h) * max(0.0, margin_ratio))
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(frame_bgr.shape[1], x + w + margin)
    y2 = min(frame_bgr.shape[0], y + h + margin)

    if x2 <= x1 or y2 <= y1:
        return frame_bgr, None
    return frame_bgr[y1:y2, x1:x2], (x1, y1, x2 - x1, y2 - y1)


def make_depth_vis(depth_map: np.ndarray, size: tuple[int, int] = (256, 256)) -> np.ndarray:
    depth = np.clip(depth_map, 0.0, 1.0)
    depth_u8 = (depth * 255.0).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)
    return cv2.resize(depth_color, size, interpolation=cv2.INTER_NEAREST)


def draw_overlay(frame: np.ndarray, pred_label: int, attack_prob: float,
                 genuine_prob: float, inference_ms: float, fps: float,
                 face_box=None):
    pred_name = LABEL_NAMES[pred_label]
    is_genuine = pred_label == 1
    color = (60, 180, 75) if is_genuine else (40, 40, 230)

    cv2.rectangle(frame, (16, 16), (470, 150), (20, 20, 20), thickness=-1)
    cv2.rectangle(frame, (16, 16), (470, 150), color, thickness=2)

    cv2.putText(
        frame,
        f'Prediction: {pred_name}',
        (30, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f'Attack prob:  {attack_prob:.4f}',
        (30, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f'Genuine prob: {genuine_prob:.4f}',
        (30, 108),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f'Infer: {inference_ms:.1f} ms | FPS: {fps:.1f}',
        (30, 138),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (200, 200, 200),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        'Press Q or ESC to quit',
        (18, frame.shape[0] - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (230, 230, 230),
        2,
        cv2.LINE_AA,
    )

    if face_box is not None:
        x, y, w, h = [int(v) for v in face_box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 215, 255), 2)
        cv2.putText(
            frame,
            'Face ROI',
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 215, 255),
            2,
            cv2.LINE_AA,
        )


def draw_no_face_overlay(frame: np.ndarray, fps: float):
    color = (0, 165, 255)
    cv2.rectangle(frame, (16, 16), (560, 110), (20, 20, 20), thickness=-1)
    cv2.rectangle(frame, (16, 16), (560, 110), color, thickness=2)
    cv2.putText(
        frame,
        'No face detected - inference skipped',
        (30, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f'FPS: {fps:.1f}',
        (30, 86),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (230, 230, 230),
        2,
        cv2.LINE_AA,
    )


class PredictionStabilizer:
    def __init__(self):
        self.genuine_prob_ema = None
        self.stable_label = None
        self.pending_label = None
        self.pending_count = 0

    def reset(self):
        self.genuine_prob_ema = None
        self.stable_label = None
        self.pending_label = None
        self.pending_count = 0

    def update(self, raw_genuine_prob: float, threshold: float):
        if self.genuine_prob_ema is None:
            self.genuine_prob_ema = raw_genuine_prob
        else:
            self.genuine_prob_ema = (
                (1.0 - PROB_SMOOTH_ALPHA) * self.genuine_prob_ema
                + PROB_SMOOTH_ALPHA * raw_genuine_prob
            )

        smooth_genuine = float(self.genuine_prob_ema)
        smooth_attack = 1.0 - smooth_genuine

        if self.stable_label is None:
            self.stable_label = 1 if smooth_genuine >= threshold else 0
            return self.stable_label, smooth_attack, smooth_genuine

        high = min(1.0, threshold + DECISION_HYSTERESIS)
        low = max(0.0, threshold - DECISION_HYSTERESIS)
        candidate = self.stable_label
        if self.stable_label == 1 and smooth_genuine <= low:
            candidate = 0
        elif self.stable_label == 0 and smooth_genuine >= high:
            candidate = 1

        if candidate != self.stable_label:
            if self.pending_label == candidate:
                self.pending_count += 1
            else:
                self.pending_label = candidate
                self.pending_count = 1
            if self.pending_count >= SWITCH_CONFIRM_FRAMES:
                self.stable_label = candidate
                self.pending_label = None
                self.pending_count = 0
        else:
            self.pending_label = None
            self.pending_count = 0

        return self.stable_label, smooth_attack, smooth_genuine


def main():
    args = parse_args()

    onnx_path = os.path.abspath(args.onnx_path)
    print('=' * 70)
    print('  FAS ONNX Camera Inference')
    print('=' * 70)
    print(f'  ONNX model: {onnx_path}')
    print(f'  Camera ID:  {args.camera_id}')
    print(f'  Threshold:  {args.threshold:.2f}')
    print(f'  Capture:    {args.width}x{args.height}')
    print(f'  Flip:       {args.flip}')
    print(f'  Show depth: {args.show_depth}')
    print(f'  Face crop:  {args.face_crop}')
    print('=' * 70)

    session = build_session(onnx_path)
    capture = open_camera(args.camera_id, args.width, args.height)
    face_detector = create_face_detector() if args.face_crop else None
    stabilizer = PredictionStabilizer()

    fps = 0.0
    last_loop_t = time.perf_counter()

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                print('[WARN] Failed to read frame from camera, stop inference.')
                break

            if args.flip:
                frame = cv2.flip(frame, 1)

            face_box = None
            infer_frame = frame
            if face_detector is not None:
                detected_faces = detect_faces(frame, face_detector)
                if len(detected_faces) != 1 and not args.allow_no_face_inference:
                    stabilizer.reset()
                    now = time.perf_counter()
                    loop_fps = 1.0 / max(now - last_loop_t, 1e-6)
                    fps = loop_fps if fps == 0.0 else 0.9 * fps + 0.1 * loop_fps
                    last_loop_t = now

                    display = frame.copy()
                    draw_no_face_overlay(display, fps)
                    cv2.imshow(WINDOW_NAME, display)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord('q'), ord('Q')):
                        break
                    continue

                detected_box = detected_faces[0] if detected_faces else None

                infer_frame, face_box = crop_face_region(
                    frame,
                    detected_box,
                    margin_ratio=args.crop_margin,
                )

            input_tensor = preprocess_bgr_frame(infer_frame)

            infer_t0 = time.perf_counter()
            cls_logits, depth_pred = run_inference(session, input_tensor)
            inference_ms = (time.perf_counter() - infer_t0) * 1000.0

            probs = softmax(cls_logits)[0]
            raw_genuine_prob = float(probs[1])
            pred_label, attack_prob, genuine_prob = stabilizer.update(
                raw_genuine_prob,
                args.threshold,
            )

            now = time.perf_counter()
            loop_fps = 1.0 / max(now - last_loop_t, 1e-6)
            fps = loop_fps if fps == 0.0 else 0.9 * fps + 0.1 * loop_fps
            last_loop_t = now

            display = frame.copy()
            draw_overlay(display, pred_label, attack_prob, genuine_prob, inference_ms, fps, face_box)
            cv2.imshow(WINDOW_NAME, display)

            if args.show_depth:
                depth_vis = make_depth_vis(depth_pred[0, 0])
                cv2.imshow(DEPTH_WINDOW_NAME, depth_vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
