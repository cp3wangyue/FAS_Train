# -*- coding: utf-8 -*-
"""
export.py - 将训练好的 PyTorch 权重导出为 ONNX 模型

用途：
    1. 加载 train.py 产出的 best_model.pth / last_model.pth
    2. 构建 DualHeadFASNet 并恢复权重
    3. 导出 ONNX 文件，供后续 onnxruntime / UI 部署使用
    4. 可选做导出后的基础校验（若环境已安装 onnx / onnxruntime）
"""

import os
import argparse
import importlib.util
from typing import Optional

import torch

from core.model import DualHeadFASNet


def parse_args():
    parser = argparse.ArgumentParser(description='Export FAS model to ONNX')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=os.path.join('D:', os.sep, 'FAS_Train', 'weights', 'best_model.pth'),
        help='Checkpoint path (.pth), default: weights/best_model.pth'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output ONNX path, default: same dir as checkpoint with .onnx suffix'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Dummy batch size for export, default: 1'
    )
    parser.add_argument(
        '--opset',
        type=int,
        default=17,
        help='ONNX opset version, default: 17'
    )
    parser.add_argument(
        '--dynamic_batch',
        action='store_true',
        help='Export with dynamic batch axis'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='If available, verify exported ONNX with onnxruntime'
    )
    return parser.parse_args()


def resolve_output_path(checkpoint_path: str, output_path: Optional[str]) -> str:
    if output_path:
        return output_path
    ckpt_dir = os.path.dirname(checkpoint_path)
    ckpt_stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
    return os.path.join(ckpt_dir, f'{ckpt_stem}.onnx')


def ensure_export_dependencies(verify: bool):
    if importlib.util.find_spec('onnx') is None:
        raise RuntimeError(
            'ONNX export requires the `onnx` package.\n'
            'Suggested command:\n'
            '  conda run -n FAS_Train pip install onnx'
        )

    if verify and importlib.util.find_spec('onnxruntime') is None:
        raise RuntimeError(
            'You enabled --verify, but `onnxruntime` is not installed.\n'
            'Suggested command:\n'
            '  conda run -n FAS_Train pip install onnxruntime'
        )


def load_model_from_checkpoint(checkpoint_path: str) -> tuple[DualHeadFASNet, dict]:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'model_state_dict' not in ckpt:
        raise KeyError('Checkpoint missing model_state_dict')

    model = DualHeadFASNet(num_classes=2, pretrained=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()
    return model, ckpt


def export_to_onnx(model: DualHeadFASNet, output_path: str, batch_size: int,
                   opset: int, dynamic_batch: bool):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dummy_input = torch.randn(batch_size, 3, 224, 224, dtype=torch.float32)
    input_names = ['rgb']
    output_names = ['cls_logits', 'depth_pred']
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            'rgb': {0: 'batch'},
            'cls_logits': {0: 'batch'},
            'depth_pred': {0: 'batch'},
        }

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

    return dummy_input


def verify_export(model: DualHeadFASNet, onnx_path: str, dummy_input: torch.Tensor):
    try:
        import onnxruntime as ort
    except ImportError:
        print('[Verify] onnxruntime not installed, skip runtime verification.')
        return

    with torch.no_grad():
        pt_cls, pt_depth = model(dummy_input)

    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    ort_inputs = {'rgb': dummy_input.cpu().numpy()}
    ort_cls, ort_depth = session.run(None, ort_inputs)

    cls_diff = float(torch.max(torch.abs(pt_cls.cpu() - torch.from_numpy(ort_cls))))
    depth_diff = float(torch.max(torch.abs(pt_depth.cpu() - torch.from_numpy(ort_depth))))

    print(f'[Verify] cls_logits max abs diff: {cls_diff:.8f}')
    print(f'[Verify] depth_pred max abs diff: {depth_diff:.8f}')


def main():
    args = parse_args()

    checkpoint_path = os.path.abspath(args.checkpoint)
    output_path = os.path.abspath(resolve_output_path(checkpoint_path, args.output))

    print('=' * 70)
    print('  FAS ONNX Export')
    print('=' * 70)
    print(f'  Checkpoint:   {checkpoint_path}')
    print(f'  Output:       {output_path}')
    print(f'  Batch size:   {args.batch_size}')
    print(f'  Opset:        {args.opset}')
    print(f'  Dynamic axes: {args.dynamic_batch}')
    print(f'  Verify:       {args.verify}')
    print('=' * 70)

    ensure_export_dependencies(args.verify)

    model, ckpt = load_model_from_checkpoint(checkpoint_path)
    print(f"[Load] epoch={ckpt.get('epoch')}  "
          f"val_acc={ckpt.get('val_acc')}  "
          f"val_acer={ckpt.get('val_acer')}")

    dummy_input = export_to_onnx(
        model=model,
        output_path=output_path,
        batch_size=args.batch_size,
        opset=args.opset,
        dynamic_batch=args.dynamic_batch,
    )

    if os.path.exists(output_path):
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f'[Export] ONNX saved: {output_path} ({file_size_mb:.2f} MB)')
    else:
        raise FileNotFoundError(f'Export failed, ONNX file not found: {output_path}')

    if args.verify:
        verify_export(model, output_path, dummy_input)

    print('[Done] Export finished successfully.')


if __name__ == '__main__':
    main()
