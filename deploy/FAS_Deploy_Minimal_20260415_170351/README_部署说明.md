# FAS 最小部署包说明

该目录是人脸呈现攻击检测系统的最小运行版，面向笔记本端 GUI 演示和实时摄像头测试。

## 目录内容

```text
FAS_Deploy_Minimal_20260415_170351/
├── ui/
│   └── app.py
├── weights/
│   └── best_model.onnx
├── infer_onnx.py
├── infer_onnx_camera.py
├── summarize_gui_runs.py
├── analyze_threshold.py
├── collect_camera_samples.py
├── requirements_deploy.txt
├── env_name.txt
├── deploy_and_run.bat
├── setup_env.bat
└── start_gui.bat
```

## 运行方式

当前部署包默认使用已经配置好依赖的 Miniconda 环境，不再自动创建 `.venv`，也不再重复安装依赖。

1. 打开 `env_name.txt`
2. 将其中内容改成笔记本上的真实 conda 环境名
3. 双击 `deploy_and_run.bat`

示例：如果环境名是 `fas`，则 `env_name.txt` 中只保留：

```text
fas
```

`deploy_and_run.bat` 会自动执行以下流程：

```text
查找 conda.bat -> 读取 env_name.txt -> 激活环境 -> 运行 ui/app.py
```

## 最小依赖

部署端只需要：

- numpy
- opencv-python
- onnxruntime
- PyQt5

部署端原则上不需要：

- torch
- torchvision
- albumentations
- onnx

## GUI 使用顺序

1. 点击“加载模型”
2. 点击“打开摄像头”
3. 点击“开始检测”
4. 根据场景调整参数

常用参数：

- `Capture Width / Height`：建议先用 `640x480`
- `Infer Every N Frames`：建议先用 `2`
- `启用低照度增强`：暗光场景开启
- `启用人脸裁剪`：默认开启，系统会检测最大人脸区域并裁剪后送入模型
- `记录实验数据 (CSV)`：建议保持开启，方便论文第 5 章统计
- `Sample Label`：采集阈值分析数据时选择 `真人 Genuine` 或 `攻击 Attack`，普通演示可保持未知

## 输出结果

每次开始检测会生成一个运行目录：

```text
weights/gui_runs/run_时间戳/
```

其中可能包含：

- `session_log.csv`：逐帧推理日志
- `frame_*.jpg`：手动保存的当前帧截图
- `depth_*.png`：手动保存的伪深度图

## 汇总实验日志

运行以下命令可以汇总所有 GUI 测试日志：

```bat
python summarize_gui_runs.py
```

默认情况下，少于 10 条记录的运行目录会被视为开发验证日志并自动忽略。正式测试建议每组运行 20 到 30 秒。

输出文件：

```text
weights/gui_runs/gui_runs_summary.csv
weights/gui_runs/gui_runs_summary.md
```

该汇总表可用于论文第 5 章端侧 FPS 与推理耗时统计。

## 分析最佳阈值

如果 GUI 测试时已经选择了 `Sample Label`，可以运行：

```bat
python analyze_threshold.py
```

输出文件：

```text
weights/gui_runs/threshold_analysis.csv
```

脚本会从带标签日志中扫描 `0.00` 到 `1.00` 的 Genuine 阈值，并输出 Accuracy、APCER、BPCER、ACER，用于选择更适合真实摄像头场景的阈值。

## 采集真实摄像头微调样本

采集真人样本：

```bat
python collect_camera_samples.py --label genuine --camera_id 0 --face_crop --max_images 300
```

采集攻击样本：

```bat
python collect_camera_samples.py --label attack --camera_id 0 --face_crop --max_images 300
```

输出目录：

```text
data/real_camera_finetune/
```

这些样本后续可用于微调模型，改善真实摄像头场景下真人被误判为攻击的问题。

## 摄像头打不开时

优先检查：

- 关闭系统相机、微信、QQ、浏览器会议等可能占用摄像头的软件
- 检查 Windows 隐私设置是否允许桌面应用访问摄像头
- 将 `Camera ID` 从 `0` 改为 `1`

## 暗光场景建议

如果界面提示 `LOW BRIGHTNESS` 或亮度较低：

- 开启“启用低照度增强”
- 适当给脸部补光
- 保持人脸靠近摄像头并位于画面中央

暗光增强可以改善输入质量，但不能保证任意光照下都稳定识别，因此论文中应作为系统优化功能和局限性分析来表述。
