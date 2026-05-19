# -*- coding: utf-8 -*-
"""
infer_onnx.py - ONNX 单图推理脚本

功能：
    1. 加载 weights/best_model.onnx
    2. 对单张 RGB 图片执行与训练一致的预处理
    3. 输出真假分类概率与阈值判定结果
    4. 可选保存 32x32 伪深度图的可视化结果

示例：
    conda run -n FAS_Train python D:/FAS_Train/infer_onnx.py ^
        --image_path D:/FAS_Train/data/CASIA-SURF/train/Training/fake_part/CLKJ_AS0005/04_en_b.rssdk/color/101.jpg ^
        --save_depth
"""

import os
import time
import argparse

import cv2
import numpy as np
import onnxruntime as ort


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 与 core/dataset.py 保持一致的预处理参数
RGB_SIZE = 224
DEPTH_SIZE = 32
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
LABEL_NAMES = {
    0: 'Attack',
    1: 'Genuine',
}


def parse_args():
    parser = argparse.ArgumentParser(description='Single-image inference with ONNX')
    parser.add_argument(
        '--onnx_path',
        type=str,
        default=os.path.join(PROJECT_ROOT, 'weights', 'best_model.onnx'),
        help='Path to ONNX model, default: weights/best_model.onnx'
    )
    parser.add_argument(
        '--image_path',
        type=str,
        required=True,
        help='Path to one RGB image for inference'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Threshold on genuine probability, default: 0.5'
    )
    parser.add_argument(
        '--save_depth',
        action='store_true',
        help='Whether to save the predicted depth map as PNG'
    )
    parser.add_argument(
        '--depth_output',
        type=str,
        default=None,
        help='Optional output path for saved depth map'
    )
    return parser.parse_args()


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def preprocess_image(image_path: str) -> tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f'Image not found: {image_path}')

    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f'Failed to read image: {image_path}')

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (RGB_SIZE, RGB_SIZE), interpolation=cv2.INTER_LINEAR)

    rgb = rgb.astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    chw = np.transpose(rgb, (2, 0, 1))
    batched = np.expand_dims(chw, axis=0).astype(np.float32)
    return batched, bgr


def build_session(onnx_path: str) -> ort.InferenceSession:
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f'ONNX model not found: {onnx_path}')

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session_options.intra_op_num_threads = max(1, min(4, os.cpu_count() or 4))
    session_options.inter_op_num_threads = 1

    available_providers = ort.get_available_providers()
    if 'CUDAExecutionProvider' in available_providers:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    elif 'DmlExecutionProvider' in available_providers:
        providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']

    return ort.InferenceSession(
        onnx_path,
        sess_options=session_options,
        providers=providers,
    )


def resolve_depth_output_path(image_path: str, depth_output: str = None) -> str:
    if depth_output:
        return os.path.abspath(depth_output)

    output_dir = os.path.join(PROJECT_ROOT, 'weights', 'infer_outputs')
    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(output_dir, f'{stem}_depth.png')


def save_depth_map(depth_pred: np.ndarray, output_path: str) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    depth = np.clip(depth_pred, 0.0, 1.0)
    depth_u8 = (depth * 255.0).astype(np.uint8)
    depth_vis = cv2.resize(depth_u8, (256, 256), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(output_path, depth_vis)
    return output_path


def run_inference(session: ort.InferenceSession, input_tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    input_name = session.get_inputs()[0].name
    output_names = [out.name for out in session.get_outputs()]
    outputs = session.run(output_names, {input_name: input_tensor})
    output_map = dict(zip(output_names, outputs))

    cls_logits = output_map.get('cls_logits', outputs[0])
    depth_pred = output_map.get('depth_pred', outputs[-1])
    return cls_logits, depth_pred


def main():
    args = parse_args()

    onnx_path = os.path.abspath(args.onnx_path)
    image_path = os.path.abspath(args.image_path)

    print('=' * 70)
    print('  FAS ONNX Single-Image Inference')
    print('=' * 70)
    print(f'  ONNX model: {onnx_path}')
    print(f'  Input image: {image_path}')
    print(f'  Threshold:   {args.threshold:.2f}')
    print(f'  Save depth:  {args.save_depth}')
    print('=' * 70)

    input_tensor, _ = preprocess_image(image_path)
    session = build_session(onnx_path)

    t0 = time.perf_counter()
    cls_logits, depth_pred = run_inference(session, input_tensor)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    probs = softmax(cls_logits)[0]
    attack_prob = float(probs[0])
    genuine_prob = float(probs[1])

    pred_label = 1 if genuine_prob >= args.threshold else 0
    pred_name = LABEL_NAMES[pred_label]

    depth_map = depth_pred[0, 0]
    depth_min = float(np.min(depth_map))
    depth_max = float(np.max(depth_map))
    depth_mean = float(np.mean(depth_map))

    print(f'[Result] Pred label:    {pred_label} ({pred_name})')
    print(f'[Result] Attack prob:   {attack_prob:.6f}')
    print(f'[Result] Genuine prob:  {genuine_prob:.6f}')
    print(f'[Result] Inference:     {elapsed_ms:.2f} ms')
    print(f'[Depth]  shape:         {depth_pred.shape}')
    print(f'[Depth]  range:         [{depth_min:.6f}, {depth_max:.6f}]')
    print(f'[Depth]  mean:          {depth_mean:.6f}')

    if args.save_depth:
        depth_output = resolve_depth_output_path(image_path, args.depth_output)
        saved_path = save_depth_map(depth_map, depth_output)
        print(f'[Depth]  saved to:      {saved_path}')


if __name__ == '__main__':
    main()
