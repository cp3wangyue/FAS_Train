# -*- coding: utf-8 -*-
"""
core/dataset.py - CASIA-SURF 人脸防伪数据集加载器

功能说明：
    1. 解析 CASIA-SURF 官方 .txt 列表文件（每行格式：RGB路径 Depth路径 IR路径 标签）
    2. 读取 RGB 图（3通道）和 Depth 图（单通道灰度）
    3. 核心逻辑：攻击样本（label=0）的深度图强制替换为全黑矩阵（伪深度监督）
    4. 使用 albumentations 保证 RGB 与 Depth 的空间变换绝对同步
    5. 输出: rgb_tensor [3, 224, 224] (ImageNet标准化), depth_tensor [1, 32, 32], label_tensor (long)
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A


class CASIASURFDataset(Dataset):
    """
    CASIA-SURF 多模态人脸防伪数据集。

    参数:
        list_file (str): 官方 .txt 列表文件的路径（如 data/CASIA-SURF/train/train_list.txt）
        transform (albumentations.Compose, optional): 数据增强管线。
            若为 None，则自动构建包含水平翻转 + 归一化的默认管线。
    """

    # ====================== RGB 与 Depth 的目标尺寸 ======================
    RGB_SIZE = 224      # RGB 图统一缩放到 224x224 送入主干网络
    DEPTH_SIZE = 32     # Depth 图统一缩放到 32x32 与网络分支输出对齐

    # ====================== ImageNet 预训练标准化参数 ======================
    # MobileNetV3 预训练权重期望输入经过该标准化
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self, list_file: str, transform=None):
        """
        初始化数据集：读取列表文件，解析每条样本的路径与标签。

        Args:
            list_file: .txt 列表文件的绝对或相对路径
            transform: albumentations 数据增强管线（可选）
        """
        super().__init__()

        # ---------- 1. 确定列表文件所在目录，后续用于拼接相对路径 ----------
        self.root_dir = os.path.dirname(list_file)

        # ---------- 2. 逐行解析列表文件 ----------
        self.samples = []   # 存储 (rgb_path, depth_path, label) 三元组

        with open(list_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue    # 跳过空行

                parts = line.split()
                # 官方格式：RGB相对路径  Depth相对路径  IR相对路径  标签
                # 我们只取 RGB、Depth 和标签，忽略 IR 通道
                rgb_rel = parts[0]
                depth_rel = parts[1]
                # parts[2] 是 IR 路径，本项目不使用，直接跳过
                label = int(parts[3])

                # 使用 os.path.join 拼接为绝对路径
                rgb_path = os.path.join(self.root_dir, rgb_rel)
                depth_path = os.path.join(self.root_dir, depth_rel)

                self.samples.append((rgb_path, depth_path, label))

        print(f"[CASIASURFDataset] 成功加载 {len(self.samples)} 条样本，"
              f"列表文件: {list_file}")

        # ---------- 3. 设置数据增强管线 ----------
        if transform is not None:
            self.transform = transform
        else:
            # 默认增强管线：训练阶段使用水平翻转 + 颜色抖动
            self.transform = self._build_default_transform()

    @staticmethod
    def _build_default_transform():
        """
        构建默认的 albumentations 数据增强管线。

        关键设计：
            - 空间变换（如水平翻转）会通过 image + mask 机制自动同步到 Depth 图
            - 颜色/亮度变换仅作用于 RGB（image），不影响 Depth（mask）
            - 此处不做 Normalize，归一化在 __getitem__ 中通过除以 255 手动完成
        """
        return A.Compose([
            # ---- 空间变换（同步作用于 RGB 和 Depth）----
            A.HorizontalFlip(p=0.5),                    # 水平翻转
            A.Affine(
                translate_percent=(-0.05, 0.05),          # 平移幅度
                scale=(0.9, 1.1),                         # 缩放幅度
                rotate=(-15, 15),                         # 旋转角度
                border_mode=cv2.BORDER_CONSTANT,          # 填充模式
                fill=0,                                   # 填充值
                p=0.5
            ),
            # ---- 颜色变换（仅作用于 RGB image，不影响 Depth mask）----
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.5
            ),
        ])

    def __len__(self):
        """返回数据集样本总数"""
        return len(self.samples)

    def __getitem__(self, index: int):
        """
        获取第 index 条样本，执行完整的数据处理流水线。

        处理步骤：
            1. 读取 RGB 图（BGR → RGB，Resize 到 224x224）
            2. 读取 Depth 图（灰度，Resize 到 32x32）
            3. 核心逻辑：攻击样本深度图强制归零
            4. 同步数据增强（albumentations image + mask 机制）
            5. 转换为 PyTorch Tensor，RGB 做 ImageNet 标准化，Depth 归一化到 [0, 1]

        Returns:
            rgb_tensor:   形状 [3, 224, 224]，float32，ImageNet 标准化后
            depth_tensor: 形状 [1, 32, 32]，float32，值域 [0, 1]
            label_tensor: 标量 tensor，torch.long 类型（0=攻击，1=真人）
        """
        rgb_path, depth_path, label = self.samples[index]

        # ==================== 步骤 1: 读取 RGB 图 ====================
        rgb_img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if rgb_img is None:
            raise FileNotFoundError(f"无法读取 RGB 图片: {rgb_path}")
        # OpenCV 默认 BGR 格式，转换为 RGB
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        # 统一缩放到 224x224
        rgb_img = cv2.resize(rgb_img, (self.RGB_SIZE, self.RGB_SIZE),
                             interpolation=cv2.INTER_LINEAR)

        # ==================== 步骤 2: 读取 Depth 图 ====================
        depth_img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        if depth_img is None:
            raise FileNotFoundError(f"无法读取 Depth 图片: {depth_path}")
        # 统一缩放到 32x32（与网络深度分支输出尺寸对齐）
        depth_img = cv2.resize(depth_img, (self.DEPTH_SIZE, self.DEPTH_SIZE),
                               interpolation=cv2.INTER_LINEAR)

        # ==================== 步骤 3: 伪深度监督核心逻辑 ====================
        # 如果是攻击样本（label=0），深度图标签强制置为全黑矩阵
        # 设计理念：攻击样本（照片/屏幕）不具备真实人脸的 3D 深度信息，
        #           因此其深度标签应为全零，迫使网络学会区分真假人脸的深度差异
        if label == 0:
            depth_img = np.zeros_like(depth_img)

        # ==================== 步骤 4: 同步数据增强 ====================
        # 关键：RGB 图尺寸为 224x224，Depth 图尺寸为 32x32，二者尺寸不同
        # albumentations 的 mask 会自动适配 image 的空间变换比例
        # 但为了保证同步增强的正确性，需要先将 Depth 临时放大到与 RGB 相同尺寸
        depth_resized_for_aug = cv2.resize(depth_img,
                                           (self.RGB_SIZE, self.RGB_SIZE),
                                           interpolation=cv2.INTER_LINEAR)

        # 执行同步增强：RGB 作为 image，Depth 作为 mask
        # mask 参数会同步接收所有空间变换，但不会被颜色变换影响
        augmented = self.transform(image=rgb_img, mask=depth_resized_for_aug)
        rgb_img = augmented['image']
        depth_augmented = augmented['mask']

        # 增强完成后，将 Depth 缩回 32x32 目标尺寸
        depth_img = cv2.resize(depth_augmented,
                               (self.DEPTH_SIZE, self.DEPTH_SIZE),
                               interpolation=cv2.INTER_LINEAR)

        # ==================== 步骤 5: 转换为 PyTorch Tensor ====================
        # --- RGB: (H, W, C) -> (C, H, W)，/255 后做 ImageNet 标准化 ---
        rgb_tensor = torch.from_numpy(
            rgb_img.astype(np.float32) / 255.0
        ).permute(2, 0, 1)  # [3, 224, 224]
        # ImageNet 标准化：匹配 MobileNetV3 预训练权重的输入分布
        mean = torch.tensor(self.IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(self.IMAGENET_STD).view(3, 1, 1)
        rgb_tensor = (rgb_tensor - mean) / std

        # --- Depth: (H, W) -> (1, H, W)，归一化到 [0, 1] ---
        depth_tensor = torch.from_numpy(
            depth_img.astype(np.float32) / 255.0
        ).unsqueeze(0)      # [1, 32, 32]

        # --- Label: 转为 long 类型的标量张量 ---
        label_tensor = torch.tensor(label, dtype=torch.long)

        return rgb_tensor, depth_tensor, label_tensor


# ========================= 单元测试入口 =========================
if __name__ == '__main__':
    """
    测试代码：验证 CASIASURFDataset 能否正确读取数据并输出预期形状的张量。
    运行方式：在项目根目录执行 python -m core.dataset
    """
    # 列表文件路径（相对于项目根目录）
    list_path = os.path.join('data', 'CASIA-SURF', 'train', 'train_list.txt')

    print("=" * 60)
    print("正在初始化 CASIASURFDataset ...")
    print("=" * 60)

    # 创建数据集实例
    dataset = CASIASURFDataset(list_file=list_path)

    print(f"\n数据集样本总数: {len(dataset)}")
    print("-" * 60)

    # 读取第一条样本并打印形状
    rgb, depth, label = dataset[0]
    print(f"[第 1 条样本]")
    print(f"  RGB   Tensor shape: {rgb.shape}    dtype: {rgb.dtype}   "
          f"值域: [{rgb.min():.4f}, {rgb.max():.4f}]")
    print(f"  Depth Tensor shape: {depth.shape}   dtype: {depth.dtype}   "
          f"值域: [{depth.min():.4f}, {depth.max():.4f}]")
    print(f"  Label Tensor value: {label.item()}       dtype: {label.dtype}")

    # 额外验证：对比一条真人样本和一条攻击样本的深度图
    print("\n" + "=" * 60)
    print("验证伪深度监督逻辑（攻击样本深度图应为全零）...")
    print("=" * 60)

    # 遍历找到一条真人和一条攻击样本
    real_idx, fake_idx = None, None
    for i, (_, _, lbl) in enumerate(dataset.samples):
        if lbl == 1 and real_idx is None:
            real_idx = i
        if lbl == 0 and fake_idx is None:
            fake_idx = i
        if real_idx is not None and fake_idx is not None:
            break

    if real_idx is not None:
        _, depth_real, lbl_real = dataset[real_idx]
        print(f"\n  真人样本 (index={real_idx}): label={lbl_real.item()}, "
              f"depth 均值={depth_real.mean():.4f} (应大于 0)")

    if fake_idx is not None:
        _, depth_fake, lbl_fake = dataset[fake_idx]
        print(f"  攻击样本 (index={fake_idx}): label={lbl_fake.item()}, "
              f"depth 均值={depth_fake.mean():.4f} (应等于 0.0000)")
        assert depth_fake.sum().item() == 0.0, \
            "❌ 错误：攻击样本的深度图未被正确归零！"
        print("  [PASS] 伪深度监督逻辑验证通过！攻击样本深度图已全零。")

    print("\n" + "=" * 60)
    print("所有测试通过！CASIASURFDataset 工作正常。")
    print("=" * 60)
