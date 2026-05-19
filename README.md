# 面向身份认证的人脸呈现攻击检测系统

本项目面向身份认证场景中的人脸呈现攻击检测任务，目标是识别真人样本与照片、屏幕回放等攻击样本，并实现可在普通笔记本环境中运行的实时检测系统。项目围绕模型训练、ONNX 导出、摄像头实时推理与 PyQt5 图形界面演示构建了较完整的实验与部署链路。

## 项目概述

- 课题名称：面向身份认证的人脸呈现攻击检测系统设计与实现
- 技术路线：`MobileNetV3-Small + Dual-Head 分类/伪深度分支 + ONNX Runtime + OpenCV + PyQt5`
- 标签定义：`0 = Attack / Spoof`，`1 = Genuine / Bona Fide`
- 项目地址：[https://github.com/cp3wangyue/FAS_Train](https://github.com/cp3wangyue/FAS_Train)

## 项目声明

- 项目名称：面向身份认证的人脸呈现攻击检测系统
- 开发语言：Python
- 框架与工具：PyTorch、ONNX Runtime、OpenCV、PyQt5
- 核心技术：人脸呈现攻击检测、轻量级卷积神经网络、伪深度监督、实时摄像头推理、桌面端图形界面

## 主要功能

- 使用 `CASIA-SURF` 数据列表完成训练数据读取与伪深度监督
- 基于 `MobileNetV3-Small` 构建双分支人脸防伪网络
- 支持训练、验证、日志记录、曲线绘制、断点续训与 ONNX 导出
- 支持单图推理、批量推理、摄像头实时推理与 PyQt5 图形界面演示
- 支持 GUI 逐帧日志记录、阈值分析与真实环境小样本微调流程

## 核心目录

```text
core/      模型、损失函数、数据集读取
ui/        PyQt5 图形界面
utils/     日志与绘图工具
deploy/    最小部署包脚本与说明
```

## 运行说明

训练与推理脚本主要使用 Python 生态完成，部署侧以 `ONNX Runtime + OpenCV + PyQt5` 为主。若要完整复现实验流程，需要准备相应数据集、模型权重与运行环境。

部署说明见 `deploy/FAS_Deploy_Minimal_20260415_170351/README_部署说明.md`。
