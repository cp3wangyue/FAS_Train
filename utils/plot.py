# -*- coding: utf-8 -*-
"""
utils/plot.py - 训练曲线绘制工具

功能：
    1. 从 training_log.csv 读取数据并绘制 Loss / Accuracy 曲线
    2. 输出高分辨率 PNG 图片，可直接粘贴到毕业论文中
    3. 支持独立运行：python -m utils.plot --log_csv <path>

曲线内容：
    - 图1 (左): Train/Val Total Loss 随 epoch 变化
    - 图2 (中): Train/Val Classification Loss + Depth Loss
    - 图3 (右): Train/Val Accuracy 随 epoch 变化
"""

import os
import csv
import argparse


def load_csv(csv_path: str) -> list:
    """从 CSV 文件加载训练日志记录（兼容新旧 CSV 格式）"""
    records = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                'epoch': int(row['epoch']),
                'lr': float(row['lr']),
                'train_loss': float(row['train_loss']),
                'train_cls_loss': float(row['train_cls_loss']),
                'train_depth_loss': float(row['train_depth_loss']),
                'train_acc': float(row['train_acc']),
                'val_loss': float(row['val_loss']),
                'val_cls_loss': float(row['val_cls_loss']),
                'val_depth_loss': float(row['val_depth_loss']),
                'val_acc': float(row['val_acc']),
                'val_apcer': float(row.get('val_apcer', 0)),
                'val_bpcer': float(row.get('val_bpcer', 0)),
                'val_acer': float(row.get('val_acer', 0)),
            })
    return records


def plot_curves(records: list, save_dir: str):
    """
    绘制训练曲线并保存为 PNG。

    Args:
        records: 字典列表，每个字典包含一个 epoch 的指标
        save_dir: 图片保存目录
    """
    # 延迟导入 matplotlib，避免无 GUI 环境报错
    import matplotlib
    matplotlib.use('Agg')       # 非交互式后端，无需显示窗口
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    epochs = [r['epoch'] for r in records]

    # ---- 通用样式 ----
    plt.rcParams.update({
        'font.size': 12,
        'figure.dpi': 150,
        'lines.linewidth': 1.5,
        'axes.grid': True,
        'grid.alpha': 0.3,
    })

    # ==================== 图 1: 总损失曲线 ====================
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [r['train_loss'] for r in records],
            'o-', markersize=3, label='Train Loss', color='#2196F3')
    ax.plot(epochs, [r['val_loss'] for r in records],
            's-', markersize=3, label='Val Loss', color='#FF5722')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Training & Validation Loss')
    ax.legend()
    fig.tight_layout()
    path1 = os.path.join(save_dir, 'loss_curve.png')
    fig.savefig(path1)
    plt.close(fig)
    print(f"  [Saved] {path1}")

    # ==================== 图 2: 分支损失曲线 ====================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 分类损失
    ax1.plot(epochs, [r['train_cls_loss'] for r in records],
             'o-', markersize=3, label='Train CLS Loss', color='#2196F3')
    ax1.plot(epochs, [r['val_cls_loss'] for r in records],
             's-', markersize=3, label='Val CLS Loss', color='#FF5722')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Classification Loss (CE)')
    ax1.set_title('Classification Branch Loss')
    ax1.legend()

    # 深度损失
    ax2.plot(epochs, [r['train_depth_loss'] for r in records],
             'o-', markersize=3, label='Train Depth Loss', color='#4CAF50')
    ax2.plot(epochs, [r['val_depth_loss'] for r in records],
             's-', markersize=3, label='Val Depth Loss', color='#FF9800')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Depth Loss (MSE)')
    ax2.set_title('Depth Branch Loss')
    ax2.legend()

    fig.tight_layout()
    path2 = os.path.join(save_dir, 'branch_loss_curve.png')
    fig.savefig(path2)
    plt.close(fig)
    print(f"  [Saved] {path2}")

    # ==================== 图 3: 准确率曲线 ====================
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [r['train_acc'] for r in records],
            'o-', markersize=3, label='Train Acc', color='#2196F3')
    ax.plot(epochs, [r['val_acc'] for r in records],
            's-', markersize=3, label='Val Acc', color='#FF5722')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Training & Validation Accuracy')
    ax.legend()
    ax.set_ylim([0, 105])
    fig.tight_layout()
    path3 = os.path.join(save_dir, 'accuracy_curve.png')
    fig.savefig(path3)
    plt.close(fig)
    print(f"  [Saved] {path3}")

    # ==================== 图 4: PAD 指标曲线 (APCER / BPCER / ACER) ====================
    has_pad = any(r.get('val_acer', 0) > 0 for r in records)
    path4 = None
    if has_pad:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, [r['val_apcer'] * 100 for r in records],
                '^-', markersize=3, label='APCER', color='#E91E63')
        ax.plot(epochs, [r['val_bpcer'] * 100 for r in records],
                'v-', markersize=3, label='BPCER', color='#9C27B0')
        ax.plot(epochs, [r['val_acer'] * 100 for r in records],
                'D-', markersize=3, label='ACER', color='#F44336')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Error Rate (%)')
        ax.set_title('PAD Metrics (APCER / BPCER / ACER)')
        ax.legend()
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        path4 = os.path.join(save_dir, 'pad_metrics_curve.png')
        fig.savefig(path4)
        plt.close(fig)
        print(f"  [Saved] {path4}")

    paths = [path1, path2, path3]
    if path4:
        paths.append(path4)
    return paths


# ========================= 命令行入口 =========================
if __name__ == '__main__':
    """
    独立运行绘图：
        python -m utils.plot --log_csv weights/training_log.csv --save_dir weights/
    """
    parser = argparse.ArgumentParser(description='FAS Training Curve Plotter')
    parser.add_argument('--log_csv', type=str,
                        default=os.path.join('weights', 'training_log.csv'),
                        help='CSV log file path')
    parser.add_argument('--save_dir', type=str,
                        default=os.path.join('weights',),
                        help='Output directory for PNG images')
    args = parser.parse_args()

    print("=" * 60)
    print("FAS Training Curve Plotter")
    print("=" * 60)

    if not os.path.exists(args.log_csv):
        print(f"[ERROR] CSV file not found: {args.log_csv}")
        print("Please run training first to generate the log file.")
        exit(1)

    records = load_csv(args.log_csv)
    print(f"Loaded {len(records)} epoch records from {args.log_csv}")

    paths = plot_curves(records, args.save_dir)
    print(f"\nDone! {len(paths)} figures saved to {args.save_dir}")
