# -*- coding: utf-8 -*-
"""
core/loss.py - 双分支联合损失函数

损失设计：
    Total Loss = cls_weight * CE_Loss(cls_logits, label)
               + depth_weight * MSE_Loss(depth_pred, depth_gt)

    - 分类分支 (CE Loss)：CrossEntropyLoss，衡量真/假二分类预测与标签的差距
    - 深度分支 (MSE Loss)：MSELoss，衡量预测伪深度图与真实深度图的像素级回归差距

    两个分支的损失通过可配置权重进行加权求和，联合优化共享 backbone。
"""

import torch
import torch.nn as nn


class DualHeadLoss(nn.Module):
    """
    双分支联合损失函数。

    将分类任务（交叉熵）与深度回归任务（均方误差）联合优化，
    通过权重系数控制两个任务对梯度更新的贡献比例。

    Args:
        cls_weight (float): 分类损失的权重系数，默认 1.0
        depth_weight (float): 深度回归损失的权重系数，默认 1.0
    """

    def __init__(self, cls_weight: float = 1.0, depth_weight: float = 1.0):
        super().__init__()

        self.cls_weight = cls_weight
        self.depth_weight = depth_weight

        # 分类分支：交叉熵损失
        # 输入为原始 logits [B, 2]，内部自动做 softmax
        self.ce_loss = nn.CrossEntropyLoss()

        # 深度分支：均方误差损失
        # 衡量预测深度图与真实深度图在每个像素上的 L2 距离
        self.mse_loss = nn.MSELoss()

    def forward(self, cls_logits: torch.Tensor, depth_pred: torch.Tensor,
                depth_gt: torch.Tensor, label: torch.Tensor):
        """
        计算联合损失。

        Args:
            cls_logits: 分类分支输出的原始 logits，形状 [B, 2]
            depth_pred: 深度分支输出的预测深度图，形状 [B, 1, 32, 32]，值域 [0, 1]
            depth_gt:   真实深度图标签，形状 [B, 1, 32, 32]，值域 [0, 1]
            label:      真/假二分类标签，形状 [B]，torch.long（0=攻击, 1=真人）

        Returns:
            total_loss (Tensor): 加权总损失（标量），用于反向传播
            cls_loss   (Tensor): 分类损失（标量），用于日志记录
            depth_loss (Tensor): 深度损失（标量），用于日志记录
        """
        # ---------- 分类损失 ----------
        cls_loss = self.ce_loss(cls_logits, label)

        # ---------- 深度回归损失 ----------
        depth_loss = self.mse_loss(depth_pred, depth_gt)

        # ---------- 加权联合损失 ----------
        total_loss = self.cls_weight * cls_loss + self.depth_weight * depth_loss

        return total_loss, cls_loss, depth_loss


# ========================= 单元测试入口 =========================
if __name__ == '__main__':
    """
    测试代码：验证 DualHeadLoss 的计算逻辑和梯度回传是否正常。
    运行方式：在项目根目录执行 python -m core.loss
    """
    print("=" * 60)
    print("正在测试 DualHeadLoss 联合损失函数...")
    print("=" * 60)

    batch_size = 4

    # ---- 模拟网络输出 ----
    # 分类 logits：随机值，需要梯度
    cls_logits = torch.randn(batch_size, 2, requires_grad=True)
    # 深度预测图：Sigmoid 后的值域 [0, 1]，需要梯度
    depth_pred = torch.sigmoid(torch.randn(batch_size, 1, 32, 32, requires_grad=True))

    # ---- 模拟真实标签 ----
    # 二分类标签：前两个为真人(1)，后两个为攻击(0)
    label = torch.tensor([1, 1, 0, 0], dtype=torch.long)
    # 深度图标签：真人有深度信息，攻击为全零（伪深度监督）
    depth_gt = torch.zeros(batch_size, 1, 32, 32)
    depth_gt[0] = torch.rand(1, 32, 32)   # 真人样本 1：有深度
    depth_gt[1] = torch.rand(1, 32, 32)   # 真人样本 2：有深度
    # depth_gt[2] 和 depth_gt[3] 保持全零（攻击样本）

    # ---- 测试默认权重 (1.0 / 1.0) ----
    print("\n--- 测试 1: 默认权重 (cls=1.0, depth=1.0) ---")
    criterion = DualHeadLoss(cls_weight=1.0, depth_weight=1.0)
    total, cls_l, depth_l = criterion(cls_logits, depth_pred, depth_gt, label)

    print(f"  分类损失 (CE):  {cls_l.item():.4f}")
    print(f"  深度损失 (MSE): {depth_l.item():.4f}")
    print(f"  总损失:         {total.item():.4f}")

    # 验证加权求和的正确性
    expected_total = 1.0 * cls_l.item() + 1.0 * depth_l.item()
    assert abs(total.item() - expected_total) < 1e-5, \
        f"[FAIL] 加权求和计算错误！期望 {expected_total:.4f}，实际 {total.item():.4f}"
    print("  [PASS] 加权求和验证通过")

    # ---- 测试自定义权重 ----
    print("\n--- 测试 2: 自定义权重 (cls=0.5, depth=2.0) ---")
    criterion_custom = DualHeadLoss(cls_weight=0.5, depth_weight=2.0)
    total_c, cls_c, depth_c = criterion_custom(cls_logits, depth_pred, depth_gt, label)

    print(f"  分类损失 (CE):  {cls_c.item():.4f}")
    print(f"  深度损失 (MSE): {depth_c.item():.4f}")
    print(f"  总损失:         {total_c.item():.4f}")

    expected_total_c = 0.5 * cls_c.item() + 2.0 * depth_c.item()
    assert abs(total_c.item() - expected_total_c) < 1e-5, \
        f"[FAIL] 自定义权重计算错误！期望 {expected_total_c:.4f}，实际 {total_c.item():.4f}"
    print("  [PASS] 自定义权重验证通过")

    # ---- 测试梯度回传 ----
    print("\n--- 测试 3: 梯度回传验证 ---")
    total.backward()
    grad_ok = cls_logits.grad is not None and cls_logits.grad.abs().sum() > 0
    print(f"  cls_logits 梯度是否存在: {cls_logits.grad is not None}")
    print(f"  cls_logits 梯度绝对值之和: {cls_logits.grad.abs().sum().item():.4f}")
    assert grad_ok, "[FAIL] 梯度回传异常，cls_logits 未收到有效梯度！"
    print("  [PASS] 梯度回传验证通过")

    print("\n" + "=" * 60)
    print("[PASS] 所有测试通过！DualHeadLoss 工作正常。")
    print("=" * 60)
