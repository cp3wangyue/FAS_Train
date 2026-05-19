# -*- coding: utf-8 -*-
"""
utils/logger.py - 训练日志记录器

功能：
    1. 将每轮训练指标写入 CSV 文件，供后续绘图和论文引用
    2. 同时输出到控制台，格式清晰对齐
    3. CSV 格式可直接被 pandas / Excel / matplotlib 读取
    4. 支持 resume 模式：追加写入已有 CSV/TXT，自动加载历史记录
"""

import os
import csv
from datetime import datetime


class TrainingLogger:
    """
    训练日志记录器，同时输出到控制台和 CSV 文件。

    CSV 列：
        epoch, lr,
        train_loss, train_cls_loss, train_depth_loss, train_acc,
        val_loss, val_cls_loss, val_depth_loss, val_acc,
        val_apcer, val_bpcer, val_acer,
        epoch_time, is_best

    Args:
        log_dir (str): 日志输出目录
        filename (str): CSV 文件名
        resume (bool): 是否为续训模式（追加而非覆盖）
    """

    CSV_HEADER = [
        'epoch', 'lr',
        'train_loss', 'train_cls_loss', 'train_depth_loss', 'train_acc',
        'val_loss', 'val_cls_loss', 'val_depth_loss', 'val_acc',
        'val_apcer', 'val_bpcer', 'val_acer',
        'epoch_time', 'is_best'
    ]

    def __init__(self, log_dir: str, filename: str = 'training_log.csv',
                 resume: bool = False):
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, filename)
        self.txt_path = os.path.join(log_dir, 'training_log.txt')
        self.records = []       # 内存副本，供绘图接口直接使用

        if resume and os.path.exists(self.csv_path):
            # ---- Resume 模式：加载已有记录到内存，后续追加写入 ----
            self._load_existing_records()
            print(f"[Logger] Resume mode: loaded {len(self.records)} "
                  f"existing epoch records from {self.csv_path}")
            # TXT 追加一条分隔线
            with open(self.txt_path, 'a', encoding='utf-8') as f:
                f.write(f"\n{'=' * 80}\n")
                f.write(f"[Resumed Training] "
                        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'=' * 80}\n\n")
        else:
            # ---- 全新训练：创建新文件 ----
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(self.CSV_HEADER)
            with open(self.txt_path, 'w', encoding='utf-8') as f:
                f.write(f"FAS Training Log - "
                        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n")

    def _load_existing_records(self):
        """从已有 CSV 文件加载历史记录到内存（用于 resume 后绘制完整曲线）"""
        self.records = []
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.records.append({
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
                    'epoch_time': float(row['epoch_time']),
                    'is_best': bool(int(row['is_best'])),
                })

    def log_config(self, config_dict: dict):
        """将训练配置写入文本日志头部"""
        with open(self.txt_path, 'a', encoding='utf-8') as f:
            f.write("Training Configuration:\n")
            for k, v in config_dict.items():
                f.write(f"  {k}: {v}\n")
            f.write("=" * 80 + "\n\n")

    def log_epoch(self, epoch: int, lr: float,
                  train_loss: float, train_cls_loss: float,
                  train_depth_loss: float, train_acc: float,
                  val_loss: float, val_cls_loss: float,
                  val_depth_loss: float, val_acc: float,
                  epoch_time: float, is_best: bool,
                  val_apcer: float = 0.0, val_bpcer: float = 0.0,
                  val_acer: float = 0.0):
        """
        记录一个 epoch 的训练指标。

        同时写入 CSV（结构化数据）和 TXT（可读日志）。
        """
        row = [
            epoch, f'{lr:.1e}',
            f'{train_loss:.4f}', f'{train_cls_loss:.4f}',
            f'{train_depth_loss:.4f}', f'{train_acc:.2f}',
            f'{val_loss:.4f}', f'{val_cls_loss:.4f}',
            f'{val_depth_loss:.4f}', f'{val_acc:.2f}',
            f'{val_apcer:.4f}', f'{val_bpcer:.4f}', f'{val_acer:.4f}',
            f'{epoch_time:.1f}', int(is_best)
        ]

        # 写入 CSV（追加）
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row)

        # 写入 TXT
        best_mark = " << BEST" if is_best else ""
        with open(self.txt_path, 'a', encoding='utf-8') as f:
            f.write(
                f"Epoch [{epoch:>3d}]  lr={lr:.1e}  time={epoch_time:.1f}s"
                f"{best_mark}\n"
                f"  [Train] loss={train_loss:.4f}  cls={train_cls_loss:.4f}  "
                f"depth={train_depth_loss:.4f}  acc={train_acc:.2f}%\n"
                f"  [Val]   loss={val_loss:.4f}  cls={val_cls_loss:.4f}  "
                f"depth={val_depth_loss:.4f}  acc={val_acc:.2f}%\n"
                f"          APCER={val_apcer:.4f}  BPCER={val_bpcer:.4f}  "
                f"ACER={val_acer:.4f}\n\n"
            )

        # 保存内存副本（数值格式，供绘图使用）
        self.records.append({
            'epoch': epoch, 'lr': lr,
            'train_loss': train_loss, 'train_cls_loss': train_cls_loss,
            'train_depth_loss': train_depth_loss, 'train_acc': train_acc,
            'val_loss': val_loss, 'val_cls_loss': val_cls_loss,
            'val_depth_loss': val_depth_loss, 'val_acc': val_acc,
            'val_apcer': val_apcer, 'val_bpcer': val_bpcer,
            'val_acer': val_acer,
            'epoch_time': epoch_time, 'is_best': is_best
        })

    def log_summary(self, best_epoch: int, best_val_acer: float,
                    save_path: str):
        """训练结束后写入汇总信息"""
        summary = (
            f"\nTraining Complete - "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"  Best Epoch: {best_epoch}\n"
            f"  Best Val ACER: {best_val_acer:.4f}\n"
            f"  Weights: {save_path}\n"
        )
        with open(self.txt_path, 'a', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(summary)
        print(summary)

    def get_records(self) -> list:
        """返回所有 epoch 记录（字典列表），供绘图接口使用"""
        return self.records
