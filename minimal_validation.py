# -*- coding: utf-8 -*-
"""
minimal_validation.py - 最小训练闭环验证脚本

验证清单：
  1. Dataset 能否正常读取一批数据
  2. Model 前向输出 shape 是否正确
  3. Loss 能否正常反向传播
  4. 能否跑通 1 个最小训练轮次（2 个 batch）
"""

import os
import sys
import torch
from torch.utils.data import DataLoader, Subset

# 确保从项目根目录导入
sys.path.insert(0, os.path.dirname(__file__))

from core.dataset import CASIASURFDataset
from core.model import DualHeadFASNet
from core.loss import DualHeadLoss


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[ENV] device = {device}")

    list_file = os.path.join('data', 'CASIA-SURF', 'train', 'train_list.txt')
    passed = 0
    total = 4

    # ============================================================
    # CHECK 1: Dataset 读取
    # ============================================================
    print("\n" + "=" * 60)
    print("[CHECK 1/4] Dataset 数据读取")
    print("=" * 60)
    try:
        dataset = CASIASURFDataset(list_file=list_file)
        # 取前 64 条做小规模验证
        mini_dataset = Subset(dataset, list(range(min(64, len(dataset)))))
        loader = DataLoader(mini_dataset, batch_size=8, shuffle=False,
                            num_workers=0)  # num_workers=0 避免多进程问题

        rgb_batch, depth_batch, label_batch = next(iter(loader))
        print(f"  RGB   batch shape: {rgb_batch.shape}   "
              f"dtype: {rgb_batch.dtype}  range: [{rgb_batch.min():.3f}, {rgb_batch.max():.3f}]")
        print(f"  Depth batch shape: {depth_batch.shape}  "
              f"dtype: {depth_batch.dtype}  range: [{depth_batch.min():.3f}, {depth_batch.max():.3f}]")
        print(f"  Label batch shape: {label_batch.shape}  "
              f"dtype: {label_batch.dtype}  values: {label_batch.tolist()}")

        assert rgb_batch.shape == (8, 3, 224, 224), f"RGB shape error: {rgb_batch.shape}"
        assert depth_batch.shape == (8, 1, 32, 32), f"Depth shape error: {depth_batch.shape}"
        assert label_batch.shape == (8,), f"Label shape error: {label_batch.shape}"
        print("  >> [CHECK 1 PASSED]")
        passed += 1
    except Exception as e:
        print(f"  >> [CHECK 1 FAILED] {e}")
        import traceback; traceback.print_exc()

    # ============================================================
    # CHECK 2: Model 前向输出 shape
    # ============================================================
    print("\n" + "=" * 60)
    print("[CHECK 2/4] Model 前向输出 shape")
    print("=" * 60)
    try:
        model = DualHeadFASNet(num_classes=2, pretrained=True).to(device)
        model.train()

        rgb_gpu = rgb_batch.to(device)
        cls_logits, depth_pred = model(rgb_gpu)

        print(f"  cls_logits shape:  {cls_logits.shape}   (expect: [8, 2])")
        print(f"  depth_pred shape:  {depth_pred.shape}  (expect: [8, 1, 32, 32])")
        print(f"  depth_pred range:  [{depth_pred.min():.4f}, {depth_pred.max():.4f}]  (expect: [0, 1])")

        assert cls_logits.shape == (8, 2)
        assert depth_pred.shape == (8, 1, 32, 32)
        assert depth_pred.min() >= 0 and depth_pred.max() <= 1
        print("  >> [CHECK 2 PASSED]")
        passed += 1
    except Exception as e:
        print(f"  >> [CHECK 2 FAILED] {e}")
        import traceback; traceback.print_exc()

    # ============================================================
    # CHECK 3: Loss 反向传播
    # ============================================================
    print("\n" + "=" * 60)
    print("[CHECK 3/4] Loss 反向传播")
    print("=" * 60)
    try:
        criterion = DualHeadLoss(cls_weight=1.0, depth_weight=1.0)
        depth_gt_gpu = depth_batch.to(device)
        label_gpu = label_batch.to(device)

        total_loss, cls_loss, depth_loss = criterion(
            cls_logits, depth_pred, depth_gt_gpu, label_gpu
        )
        print(f"  total_loss:  {total_loss.item():.4f}")
        print(f"  cls_loss:    {cls_loss.item():.4f}")
        print(f"  depth_loss:  {depth_loss.item():.4f}")

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # 验证梯度确实存在
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.parameters() if p.requires_grad)
        assert has_grad, "No gradient found!"
        print(f"  gradient check: OK")
        print("  >> [CHECK 3 PASSED]")
        passed += 1
    except Exception as e:
        print(f"  >> [CHECK 3 FAILED] {e}")
        import traceback; traceback.print_exc()

    # ============================================================
    # CHECK 4: 跑通 1 个最小训练轮次 (2 batches)
    # ============================================================
    print("\n" + "=" * 60)
    print("[CHECK 4/4] Mini-epoch (2 batches train + 1 batch val)")
    print("=" * 60)
    try:
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        train_losses = []
        correct = 0
        total_samples = 0

        for batch_i, (rgb, depth_gt, label) in enumerate(loader):
            if batch_i >= 2:
                break  # 只跑 2 个 batch

            rgb = rgb.to(device)
            depth_gt = depth_gt.to(device)
            label = label.to(device)

            cls_logits, depth_pred = model(rgb)
            total_loss, cls_loss, depth_loss = criterion(
                cls_logits, depth_pred, depth_gt, label
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            preds = cls_logits.argmax(dim=1)
            correct += (preds == label).sum().item()
            total_samples += label.size(0)
            train_losses.append(total_loss.item())

            print(f"  Train batch {batch_i+1}: loss={total_loss.item():.4f}  "
                  f"cls={cls_loss.item():.4f}  depth={depth_loss.item():.4f}")

        train_acc = 100.0 * correct / total_samples if total_samples > 0 else 0
        print(f"  Train accuracy: {train_acc:.1f}% ({correct}/{total_samples})")

        # 简单 val 验证
        model.eval()
        with torch.no_grad():
            rgb, depth_gt, label = next(iter(loader))
            rgb, depth_gt, label = rgb.to(device), depth_gt.to(device), label.to(device)
            cls_logits, depth_pred = model(rgb)
            val_loss, _, _ = criterion(cls_logits, depth_pred, depth_gt, label)
            val_preds = cls_logits.argmax(dim=1)
            val_acc = 100.0 * (val_preds == label).sum().item() / label.size(0)
            print(f"  Val   batch:   loss={val_loss.item():.4f}  acc={val_acc:.1f}%")

        print("  >> [CHECK 4 PASSED]")
        passed += 1
    except Exception as e:
        print(f"  >> [CHECK 4 FAILED] {e}")
        import traceback; traceback.print_exc()

    # ============================================================
    # 汇总
    # ============================================================
    print("\n" + "=" * 60)
    print(f"  RESULT: {passed}/{total} checks passed")
    print("=" * 60)

    if passed == total:
        print("  All checks passed! Training pipeline is ready.")
    else:
        print("  Some checks failed. Please review the errors above.")
        sys.exit(1)


if __name__ == '__main__':
    main()
