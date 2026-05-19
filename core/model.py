# -*- coding: utf-8 -*-
"""
core/model.py - 基于 MobileNetV3 的双分支人脸防伪网络

网络架构概览：
    ┌──────────────────────────────────┐
    │   Input: RGB [B, 3, 224, 224]   │
    └──────────────┬───────────────────┘
                   │
    ┌──────────────▼───────────────────┐
    │   MobileNetV3-Small Backbone    │
    │   (预训练特征提取, 去掉分类头)    │
    │   输出特征: [B, 576, 7, 7]      │
    └──────────────┬───────────────────┘
                   │
          ┌────────┴────────┐
          │                 │
    ┌─────▼─────┐     ┌────▼──────┐
    │ 分类分支   │     │ 深度分支   │
    │ GAP+FC    │     │ 反卷积上采  │
    │ → [B, 2]  │     │ → [B,1,32,32]│
    └───────────┘     └───────────┘

设计要点：
    1. 选用 MobileNetV3-Small（而非 Large）：参数量更小、推理更快，
       契合项目"轻薄本 CPU 端侧 ≥15 FPS"的部署需求。
    2. 分类分支使用 GlobalAvgPool + Dropout + FC，输出 2 类 logits（非 softmax）。
    3. 深度分支通过多层反卷积（ConvTranspose2d）将 7×7 特征图逐步上采样到 32×32，
       最后用 Sigmoid 将输出限制在 [0, 1] 范围内，与归一化后的深度标签对齐。
    4. 结构保持简洁，不过早引入注意力机制，便于后续渐进式扩展。
"""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class DualHeadFASNet(nn.Module):
    """
    双分支人脸防伪网络（Dual-Head Face Anti-Spoofing Network）。

    基于 MobileNetV3-Small 预训练 backbone，输出两个分支：
        - 分类分支 (cls_head): 真/假二分类 logits
        - 深度分支 (depth_head): 32×32 伪深度图预测

    Args:
        num_classes (int): 分类数量，默认为 2（真人 / 攻击）
        pretrained (bool): 是否加载 ImageNet 预训练权重，默认 True
        dropout_rate (float): 分类分支的 Dropout 概率，默认 0.2
    """

    # MobileNetV3-Small 最后一层卷积输出的通道数
    BACKBONE_OUT_CHANNELS = 576

    def __init__(self, num_classes: int = 2, pretrained: bool = True,
                 dropout_rate: float = 0.2):
        super().__init__()

        # ==================== 1. 构建 Backbone ====================
        # 加载 MobileNetV3-Small，仅保留特征提取部分（features 模块）
        if pretrained:
            backbone_full = mobilenet_v3_small(
                weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1
            )
        else:
            backbone_full = mobilenet_v3_small(weights=None)

        # MobileNetV3 的 features 模块：
        # 输入 [B, 3, 224, 224] → 输出 [B, 576, 7, 7]
        self.backbone = backbone_full.features

        # ==================== 2. 分类分支 (Classification Head) ====================
        # GlobalAvgPool → Dropout → 全连接层
        # 输入: [B, 576, 7, 7] → 输出: [B, num_classes]
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),    # [B, 576, 7, 7] → [B, 576, 1, 1]
            nn.Flatten(),               # [B, 576, 1, 1] → [B, 576]
            nn.Dropout(p=dropout_rate), # 防止过拟合
            nn.Linear(self.BACKBONE_OUT_CHANNELS, num_classes)  # [B, 576] → [B, 2]
        )

        # ==================== 3. 深度分支 (Depth Head) ====================
        # 通过多层反卷积将 7×7 特征图上采样到 32×32
        # 上采样路径: 7×7 → 14×14 → 28×28 → 32×32（最后一步微调尺寸）
        self.depth_head = nn.Sequential(
            # --- Stage 1: 7×7 → 14×14 ---
            # 降维: 576 → 128（减少计算量）
            nn.ConvTranspose2d(
                in_channels=self.BACKBONE_OUT_CHANNELS,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),  # [B, 576, 7, 7] → [B, 128, 14, 14]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # --- Stage 2: 14×14 → 28×28 ---
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),  # [B, 128, 14, 14] → [B, 64, 28, 28]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # --- Stage 3: 28×28 → 32×32 ---
            # 使用普通卷积 + 精确上采样的方式从 28 调整到 32
            nn.Conv2d(
                in_channels=64,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1
            ),  # [B, 64, 28, 28] → [B, 1, 28, 28]
            nn.Upsample(size=(32, 32), mode='bilinear', align_corners=False),
            # [B, 1, 28, 28] → [B, 1, 32, 32]

            # 将输出限制在 [0, 1]，与归一化后的深度标签对齐
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        """
        前向传播。

        Args:
            x: 输入 RGB 图像张量，形状 [B, 3, 224, 224]

        Returns:
            cls_logits: 分类 logits，形状 [B, 2]（未经 softmax）
            depth_pred: 预测伪深度图，形状 [B, 1, 32, 32]，值域 [0, 1]
        """
        # 提取共享特征
        features = self.backbone(x)     # [B, 3, 224, 224] → [B, 576, 7, 7]

        # 分类分支
        cls_logits = self.cls_head(features)    # [B, 576, 7, 7] → [B, 2]

        # 深度分支
        depth_pred = self.depth_head(features)  # [B, 576, 7, 7] → [B, 1, 32, 32]

        return cls_logits, depth_pred


# ========================= 单元测试入口 =========================
if __name__ == '__main__':
    """
    测试代码：验证 DualHeadFASNet 的输入输出形状是否符合预期。
    运行方式：在项目根目录执行 python -m core.model
    """
    print("=" * 60)
    print("正在构建 DualHeadFASNet（MobileNetV3-Small backbone）...")
    print("=" * 60)

    # 实例化模型（加载预训练权重）
    model = DualHeadFASNet(num_classes=2, pretrained=True)
    model.eval()  # 切换到评估模式（关闭 Dropout 和 BN 的训练行为）

    # 构造模拟输入：batch_size=4, 3通道 224x224 RGB 图
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    print(f"\n模拟输入形状: {dummy_input.shape}")

    # 前向传播
    with torch.no_grad():
        cls_logits, depth_pred = model(dummy_input)

    print("-" * 60)
    print(f"[分类分支] cls_logits  shape: {cls_logits.shape}   "
          f"(预期: [{batch_size}, 2])")
    print(f"[深度分支] depth_pred  shape: {depth_pred.shape}  "
          f"(预期: [{batch_size}, 1, 32, 32])")
    print(f"[深度分支] 值域: [{depth_pred.min():.4f}, {depth_pred.max():.4f}]  "
          f"(预期: [0, 1])")

    # 形状断言验证
    assert cls_logits.shape == (batch_size, 2), \
        f"[FAIL] 分类分支输出形状错误！期望 ({batch_size}, 2)，实际 {cls_logits.shape}"
    assert depth_pred.shape == (batch_size, 1, 32, 32), \
        f"[FAIL] 深度分支输出形状错误！期望 ({batch_size}, 1, 32, 32)，实际 {depth_pred.shape}"
    assert depth_pred.min() >= 0 and depth_pred.max() <= 1, \
        f"[FAIL] 深度分支输出值域错误！期望 [0, 1]，实际 [{depth_pred.min():.4f}, {depth_pred.max():.4f}]"

    print("\n[PASS] 所有形状与值域验证通过！")

    # 统计模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数统计:")
    print(f"  总参数量:     {total_params:>10,d} ({total_params / 1e6:.2f}M)")
    print(f"  可训练参数量: {trainable_params:>10,d} ({trainable_params / 1e6:.2f}M)")

    print("\n" + "=" * 60)
    print("DualHeadFASNet 构建与验证完毕！")
    print("=" * 60)
