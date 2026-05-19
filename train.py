# -*- coding: utf-8 -*-
"""
train.py - 人脸防伪模型训练主入口（点火脚本）

训练流程：
    1. 从 train_list.txt 加载全部样本，按受试者 ID 分组做 8:2 划分（防止数据泄漏）
    2. 训练集使用默认数据增强（翻转、仿射、颜色抖动），验证集不做增强
    3. 每个 epoch 完成 train + val 两阶段循环
    4. 记录总损失、分类/深度损失、Accuracy、APCER/BPCER/ACER
    5. 根据验证集 ACER 保存最优权重到 weights/ 目录
    6. 训练结束后自动绘制 loss/accuracy/ACER 曲线图（PNG）
    7. 支持从 checkpoint 断点续训（--resume_from），日志连续追加
    8. 首次训练时保存 subject-level split manifest 到 save_dir
"""

import os
import sys
import json
import time
import random
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
import albumentations as A

from core.dataset import CASIASURFDataset
from core.model import DualHeadFASNet
from core.loss import DualHeadLoss
from utils.logger import TrainingLogger
from utils.plot import plot_curves


# ========================= 命令行参数 =========================

def parse_args():
    parser = argparse.ArgumentParser(description='FAS Training')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint for resume training '
                             '(e.g. weights/last_model.pth)')
    return parser.parse_args()


# ========================= 训练配置区 =========================
class Config:
    """
    训练超参数配置（集中管理，便于调参）。

    调参建议：
        - batch_size: 显存不足时减小到 16，充裕时可加大到 64
        - learning_rate: baseline 用 1e-3，微调阶段建议降到 1e-4
        - cls_weight / depth_weight: 若分类效果差，可提高 cls_weight
        - num_epochs: baseline 建议 50 轮观察收敛趋势
    """

    # ---- 数据路径 ----
    list_file = os.path.join('data', 'CASIA-SURF', 'train', 'train_list.txt')
    val_ratio = 0.2             # 验证集占比（train_list 内部 8:2 划分）
    random_seed = 42            # 固定随机种子，保证划分与训练结果可复现

    # ---- 训练超参 ----
    num_epochs = 50             # 总训练轮数
    batch_size = 32             # 批次大小（RTX 4070 12GB 可稳定跑 32）
    learning_rate = 1e-3        # 初始学习率（Adam 优化器）
    weight_decay = 1e-4         # L2 正则化系数
    num_workers = 4             # DataLoader 数据加载线程数

    # ---- 学习率调度 ----
    lr_patience = 5             # 验证损失连续 N 轮不下降后衰减
    lr_factor = 0.5             # 衰减倍率

    # ---- 损失权重 ----
    cls_weight = 1.0            # 分类损失权重
    depth_weight = 1.0          # 深度损失权重

    # ---- 输出路径 ----
    save_dir = os.path.join('D:', os.sep, 'FAS_Train', 'weights')
    best_model_name = 'best_model.pth'
    last_model_name = 'last_model.pth'    # 末轮权重（用于断点续训）

    def to_dict(self) -> dict:
        """将配置导出为字典（写入日志用）"""
        return {k: v for k, v in vars(self.__class__).items()
                if not k.startswith('_') and not callable(v)}


# ========================= 工具函数 =========================

def set_seed(seed: int):
    """固定所有随机种子，确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_val_transform():
    """构建验证集增强管线（无增强，仅占位）"""
    return A.Compose([])


def _extract_subject_id(rgb_rel_path: str) -> str:
    """
    从 RGB 相对路径中提取受试者 ID。

    路径格式: Training/fake_part/CLKJ_AS0005/04_en_b.rssdk/color/101.jpg
              Training/real_part/CLKJ_CS0110/real.rssdk/color/91.jpg
    受试者 ID: CLKJ_AS0005, CLKJ_CS0110（第 3 层目录）
    """
    parts = rgb_rel_path.replace('\\', '/').split('/')
    return parts[2]


def split_dataset(list_file: str, val_ratio: float, seed: int,
                  save_dir: str = None):
    """
    按受试者 ID 分组划分训练集和验证集（防止数据泄漏）。

    策略：
        1. 解析列表文件，按受试者 ID 对所有样本索引分组
        2. 对受试者列表随机打乱（固定种子），按比例切分
        3. 同一受试者的所有帧只会出现在 train 或 val 中，不会跨集
        4. 创建两个 Dataset 实例（不同 transform），用 Subset 按索引划分
        5. 若指定 save_dir，将切分结果保存为 JSON manifest
    """
    # 先读取列表文件，提取每条样本的受试者 ID
    with open(list_file, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]

    # 按受试者分组：subject_id -> [样本索引列表]
    subject_to_indices = defaultdict(list)
    for idx, line in enumerate(lines):
        rgb_rel = line.split()[0]
        subj_id = _extract_subject_id(rgb_rel)
        subject_to_indices[subj_id].append(idx)

    # 打乱受试者列表（固定种子）
    subject_ids = sorted(subject_to_indices.keys())
    rng = random.Random(seed)
    rng.shuffle(subject_ids)

    # 按受试者数量比例切分
    val_subj_count = int(len(subject_ids) * val_ratio)
    val_subject_list = subject_ids[:val_subj_count]
    train_subject_list = subject_ids[val_subj_count:]
    val_subjects = set(val_subject_list)
    train_subjects = set(train_subject_list)

    # 根据分组收集样本索引
    train_indices = []
    val_indices = []
    for subj_id in train_subject_list:
        train_indices.extend(subject_to_indices[subj_id])
    for subj_id in val_subject_list:
        val_indices.extend(subject_to_indices[subj_id])

    # 创建两个 Dataset 实例（不同 transform）
    train_dataset = CASIASURFDataset(list_file=list_file, transform=None)
    val_dataset = CASIASURFDataset(list_file=list_file,
                                   transform=build_val_transform())

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    print(f"[Data] Subject-level split: {len(subject_ids)} subjects "
          f"(train: {len(train_subjects)}, val: {len(val_subjects)})")
    print(f"[Data] Sample count: total={len(lines)} | "
          f"train={len(train_indices)} | val={len(val_indices)}")

    # ---- 保存切分 manifest ----
    if save_dir:
        manifest_path = os.path.join(save_dir, 'split_manifest.json')
        manifest = {
            'list_file': os.path.abspath(list_file),
            'random_seed': seed,
            'val_ratio': val_ratio,
            'total_subjects': len(subject_ids),
            'total_samples': len(lines),
            'train_subjects': sorted(train_subject_list),
            'val_subjects': sorted(val_subject_list),
            'train_sample_count': len(train_indices),
            'val_sample_count': len(val_indices),
        }
        os.makedirs(save_dir, exist_ok=True)
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"[Data] Split manifest saved to: {manifest_path}")

    return train_subset, val_subset


# ========================= PAD 指标计算 =========================

def compute_pad_metrics(all_preds: list, all_labels: list):
    """
    计算 PAD（Presentation Attack Detection）标准指标。

    标签约定（CASIA-SURF）：
        - label=1: 真人 (Bona Fide / Genuine)
        - label=0: 攻击 (Presentation Attack / Spoof)

    指标定义（ISO 30107-3）：
        - APCER (Attack Presentation Classification Error Rate):
            攻击样本被错误分类为真人的比率 = FN_attack / N_attack
        - BPCER (Bona Fide Presentation Classification Error Rate):
            真人样本被错误分类为攻击的比率 = FN_genuine / N_genuine
        - ACER (Average Classification Error Rate):
            ACER = (APCER + BPCER) / 2

    Confusion Matrix:
        预测\\真实 | Attack(0) | Genuine(1)
        ---------|-----------|----------
        Attack(0)|    TN     |    FP    (BPCER = FP/N_genuine)
        Genuine(1)|   FN     |    TP    (APCER = FN/N_attack)

    Args:
        all_preds: 预测标签列表（0 或 1）
        all_labels: 真实标签列表（0 或 1）

    Returns:
        metrics: dict 包含 accuracy, apcer, bpcer, acer, confusion_matrix
    """
    preds = np.array(all_preds)
    labels = np.array(all_labels)

    # 混淆矩阵元素
    tp = int(((preds == 1) & (labels == 1)).sum())  # 真人正确
    tn = int(((preds == 0) & (labels == 0)).sum())  # 攻击正确
    fp = int(((preds == 1) & (labels == 0)).sum())  # 攻击误判为真人
    fn = int(((preds == 0) & (labels == 1)).sum())  # 真人误判为攻击

    n_attack = int((labels == 0).sum())     # 攻击样本总数
    n_genuine = int((labels == 1).sum())    # 真人样本总数
    total = len(labels)

    accuracy = (tp + tn) / total if total > 0 else 0.0
    apcer = fp / n_attack if n_attack > 0 else 0.0     # 攻击漏检率
    bpcer = fn / n_genuine if n_genuine > 0 else 0.0    # 真人误拒率
    acer = (apcer + bpcer) / 2.0

    return {
        'accuracy': accuracy,
        'apcer': apcer,
        'bpcer': bpcer,
        'acer': acer,
        'confusion_matrix': {
            'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
            'N_attack': n_attack, 'N_genuine': n_genuine,
        }
    }


# ========================= Train / Val 循环 =========================

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    训练一个 epoch。

    Returns:
        avg_total_loss, avg_cls_loss, avg_depth_loss, accuracy(%)
    """
    model.train()
    running_total, running_cls, running_depth = 0.0, 0.0, 0.0
    correct, total_samples = 0, 0

    for rgb, depth_gt, label in dataloader:
        rgb, depth_gt, label = (rgb.to(device), depth_gt.to(device),
                                label.to(device))

        cls_logits, depth_pred = model(rgb)
        total_loss, cls_loss, depth_loss = criterion(
            cls_logits, depth_pred, depth_gt, label
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        bs = label.size(0)
        running_total += total_loss.item() * bs
        running_cls += cls_loss.item() * bs
        running_depth += depth_loss.item() * bs
        correct += (cls_logits.argmax(1) == label).sum().item()
        total_samples += bs

    n = total_samples
    return running_total/n, running_cls/n, running_depth/n, 100.0*correct/n


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """
    验证一个 epoch（无梯度），同时收集 PAD 指标。

    Returns:
        avg_total_loss, avg_cls_loss, avg_depth_loss, pad_metrics(dict)
    """
    model.eval()
    running_total, running_cls, running_depth = 0.0, 0.0, 0.0
    total_samples = 0
    all_preds = []
    all_labels = []

    for rgb, depth_gt, label in dataloader:
        rgb, depth_gt, label = (rgb.to(device), depth_gt.to(device),
                                label.to(device))

        cls_logits, depth_pred = model(rgb)
        total_loss, cls_loss, depth_loss = criterion(
            cls_logits, depth_pred, depth_gt, label
        )

        bs = label.size(0)
        running_total += total_loss.item() * bs
        running_cls += cls_loss.item() * bs
        running_depth += depth_loss.item() * bs
        total_samples += bs

        # 收集所有预测和标签，用于 PAD 指标计算
        preds = cls_logits.argmax(1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(label.cpu().tolist())

    n = total_samples
    pad_metrics = compute_pad_metrics(all_preds, all_labels)

    return running_total/n, running_cls/n, running_depth/n, pad_metrics


# ========================= 主训练流程 =========================

def main():
    args = parse_args()
    cfg = Config()

    # Resume 参数校验：指定了文件但不存在时直接报错
    if args.resume_from is not None and not os.path.exists(args.resume_from):
        print(f"[ERROR] --resume_from checkpoint not found: {args.resume_from}")
        print("Please check the path or remove --resume_from to train from scratch.")
        sys.exit(1)
    is_resume = args.resume_from is not None

    # ==================== 0. 环境初始化 ====================
    set_seed(cfg.random_seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(cfg.save_dir, exist_ok=True)

    # 初始化日志记录器（resume 模式追加，否则新建）
    logger = TrainingLogger(log_dir=cfg.save_dir, resume=is_resume)
    logger.log_config(cfg.to_dict())

    print("=" * 70)
    print("  FAS Dual-Head Anti-Spoofing Training - Baseline")
    print("=" * 70)
    print(f"  Device:     {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == 'cuda' else ""))
    print(f"  Epochs:     {cfg.num_epochs}")
    print(f"  Batch size: {cfg.batch_size}")
    print(f"  LR:         {cfg.learning_rate}  (decay x{cfg.lr_factor} "
          f"after {cfg.lr_patience} stale epochs)")
    print(f"  Loss wt:    cls={cfg.cls_weight}  depth={cfg.depth_weight}")
    print(f"  Save dir:   {cfg.save_dir}")
    print(f"  Resume:     {args.resume_from if is_resume else 'None (training from scratch)'}")
    print("=" * 70)

    # ==================== 1. 数据准备 ====================
    train_subset, val_subset = split_dataset(
        cfg.list_file, cfg.val_ratio, cfg.random_seed,
        save_dir=cfg.save_dir
    )

    train_loader = DataLoader(
        train_subset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=False
    )
    print(f"[DataLoader] Train batches: {len(train_loader)}  |  "
          f"Val batches: {len(val_loader)}")

    # ==================== 2. 模型 / 损失 / 优化器 ====================
    model = DualHeadFASNet(num_classes=2, pretrained=True).to(device)
    criterion = DualHeadLoss(cls_weight=cfg.cls_weight,
                             depth_weight=cfg.depth_weight)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=cfg.lr_factor,
        patience=cfg.lr_patience
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] DualHeadFASNet params: {total_params:,d} ({total_params/1e6:.2f}M)")

    # ==================== 2.5 Resume 断点续训 ====================
    start_epoch = 1
    best_val_acer = float('inf')    # ACER 越低越好，初始设为正无穷
    best_epoch = 0

    if is_resume:
        print(f"[Resume] Loading checkpoint: {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location=device,
                          weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_acer = ckpt.get('best_val_acer', float('inf'))
        best_epoch = ckpt.get('epoch', 0)
        print(f"[Resume] Restored epoch={ckpt['epoch']}, "
              f"best_val_acer={best_val_acer:.4f}, "
              f"will start from epoch {start_epoch}")

    print("-" * 70)

    # ==================== 3. 训练循环 ====================
    best_path = os.path.join(cfg.save_dir, cfg.best_model_name)
    last_path = os.path.join(cfg.save_dir, cfg.last_model_name)

    for epoch in range(start_epoch, cfg.num_epochs + 1):
        t0 = time.time()

        # ---- Train ----
        tr_loss, tr_cls, tr_dep, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        # ---- Val (with PAD metrics) ----
        va_loss, va_cls, va_dep, pad = validate(
            model, val_loader, criterion, device
        )
        va_acc = pad['accuracy'] * 100.0
        va_apcer = pad['apcer']
        va_bpcer = pad['bpcer']
        va_acer = pad['acer']

        # ---- LR schedule ----
        scheduler.step(va_loss)
        lr_now = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        # ---- Best checkpoint（以 ACER 最小为准）----
        is_best = va_acer < best_val_acer
        if is_best:
            best_val_acer = va_acer
            best_epoch = epoch

        # 构建 checkpoint 字典
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_acc': va_acc,
            'val_loss': va_loss,
            'val_acer': va_acer,
            'best_val_acer': best_val_acer,
            'config': cfg.to_dict(),
        }

        # 保存 best 权重（ACER 刷新最低时）
        if is_best:
            torch.save(ckpt, best_path)

        # 每轮保存 last 权重（用于断点续训）
        torch.save(ckpt, last_path)

        # ---- 日志 ----
        logger.log_epoch(
            epoch=epoch, lr=lr_now,
            train_loss=tr_loss, train_cls_loss=tr_cls,
            train_depth_loss=tr_dep, train_acc=tr_acc,
            val_loss=va_loss, val_cls_loss=va_cls,
            val_depth_loss=va_dep, val_acc=va_acc,
            epoch_time=elapsed, is_best=is_best,
            val_apcer=va_apcer, val_bpcer=va_bpcer, val_acer=va_acer,
        )

        # 控制台输出
        cm = pad['confusion_matrix']
        best_tag = "  << BEST" if is_best else ""
        print(
            f"Epoch [{epoch:>3d}/{cfg.num_epochs}]  "
            f"{elapsed:.1f}s  lr={lr_now:.1e}\n"
            f"  Train | loss={tr_loss:.4f}  cls={tr_cls:.4f}  "
            f"depth={tr_dep:.4f}  acc={tr_acc:.2f}%\n"
            f"  Val   | loss={va_loss:.4f}  cls={va_cls:.4f}  "
            f"depth={va_dep:.4f}  acc={va_acc:.2f}%{best_tag}\n"
            f"        | APCER={va_apcer:.4f}  BPCER={va_bpcer:.4f}  "
            f"ACER={va_acer:.4f}\n"
            f"        | CM: TP={cm['TP']} TN={cm['TN']} "
            f"FP={cm['FP']} FN={cm['FN']}"
        )
        print("-" * 70)

    # ==================== 4. 训练结束 ====================
    logger.log_summary(best_epoch, best_val_acer, best_path)

    print("=" * 70)
    print(f"  Best val ACER: {best_val_acer:.4f} at epoch {best_epoch}")
    print(f"  Best weights: {best_path}")
    print(f"  Last weights: {last_path}")
    print(f"  Log CSV:      {logger.csv_path}")
    print(f"  Log TXT:      {logger.txt_path}")
    print("=" * 70)

    # ---- 自动绘制训练曲线 ----
    print("\nPlotting training curves...")
    try:
        records = logger.get_records()
        saved_figs = plot_curves(records, cfg.save_dir)
        print(f"Done! {len(saved_figs)} figures saved to {cfg.save_dir}")
    except Exception as e:
        print(f"[WARN] Curve plotting failed (non-critical): {e}")
        print("You can manually plot later: python -m utils.plot")


if __name__ == '__main__':
    main()