# AI 协同开发全局上下文同步文档 (FAS_Project_Context)

本文件是 `D:/FAS_Train` 项目的最新全局开发说明。后续继续写代码、补实验、部署、写论文或答辩材料时，应优先以本文件为准。

## 一、项目基础信息

- 课题名称：面向身份认证的人脸呈现攻击检测系统设计与实现
- 项目目标：面向身份认证场景，实现一个基于轻量级卷积神经网络的人脸呈现攻击检测系统，用于区分真人样本与照片/屏幕等呈现攻击样本。
- 当前技术路线：`MobileNetV3-Small + Dual-Head 分类/伪深度分支 + ONNX Runtime + OpenCV + PyQt5`
- 统一标签口径：
  - `0 = 攻击样本 (Attack / Spoof)`
  - `1 = 真人样本 (Genuine / Bona Fide)`

## 二、当前真实开发状态

截至当前仓库状态，项目已经完成从训练到端侧 GUI 演示的主要闭环：

```text
数据集读取 -> 模型训练 -> 日志/曲线/权重保存 -> ONNX 导出
-> 单图推理 -> 批量推理 -> 摄像头实时推理 -> PyQt5 GUI
-> 实验日志保存/汇总 -> 真实摄像头样本微调 -> 最小部署包
```

### 1. 已完成模块

- 数据集读取与伪深度监督：`core/dataset.py`
- 网络结构：`core/model.py`
- 联合损失函数：`core/loss.py`
- 训练、验证、日志、曲线、checkpoint 与断点续训：`train.py`
- 训练日志与曲线工具：`utils/logger.py`、`utils/plot.py`
- ONNX 导出：`export.py`
- ONNX 单图推理：`infer_onnx.py`
- ONNX 批量推理：`infer_onnx_batch.py`
- 摄像头实时推理：`infer_onnx_camera.py`
- PyQt5 桌面 GUI：`ui/app.py`
- GUI 实验日志汇总：`summarize_gui_runs.py`
- GUI 带标签日志阈值分析：`analyze_threshold.py`
- 真实摄像头微调样本采集：`collect_camera_samples.py`
- 个人真实环境小样本微调：`finetune_me.py`
- 最小部署包：`deploy/FAS_Deploy_Minimal_20260415_170351/`

### 2. GUI 当前能力

`ui/app.py` 已实现：

- ONNX 模型选择与加载
- 摄像头打开与关闭
- 实时开始/停止检测
- Attack/Genuine 预测结果显示
- Attack prob、Genuine prob、Display FPS、Infer FPS、推理耗时、Brightness 显示
- 低照度增强：CLAHE + Gamma 校正
- 跳帧推理：`Infer Every N Frames`
- 分辨率控制：`Capture Width / Height`
- OpenCV Haar cascade 轻量人脸 ROI 裁剪
- 人脸框短时复用与平滑，降低检测框跳动对推理输入的影响
- 实时预测稳定器：对 Genuine 概率做指数滑动平均，并使用滞回阈值与连续帧确认，降低 Attack/Genuine 来回跳动
- 默认部署参数：`Threshold=0.60`，`Face Crop Margin=0.80`，用于配合上下文微调模型保留屏幕边框、反光等攻击线索
- 测试样本标签记录：`未知 / 真人 / 攻击`
- 伪深度图显示
- 实验数据 CSV 自动记录
- 当前帧截图与伪深度图保存
- 结果目录快速打开

### 3. 已完成的真实摄像头测试

已使用 USB 连接手机作为电脑摄像头完成 3 组真实 GUI 对照测试：

| 运行目录 | 记录数 | 分辨率 | 跳帧 | 低照度增强 | 人脸裁剪 | 人脸检出率 | Display FPS | Infer FPS | 推理耗时(ms) | 亮度 |
| --- | ---: | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `run_20260420_200637` | 283 | 640x480 | 2 | 关闭 | 开启 | 0.6855 | 21.48 | 13.99 | 80.52 | 140.62 |
| `run_20260420_201130` | 69 | 640x480 | 2 | 开启 | 开启 | 0.9565 | 21.16 | 11.40 | 96.40 | 148.33 |
| `run_20260420_201157` | 112 | 640x480 | 2 | 开启 | 关闭 | 0.0000 | 21.73 | 14.09 | 79.58 | 148.55 |

截图与深度图：

- `weights/gui_runs/run_20260420_200637/frame_20260420_200715_450.jpg`
- `weights/gui_runs/run_20260420_200637/depth_20260420_200715_450.png`

这些结果可以用于论文第 5 章证明系统在当前配置下具备实时演示能力。若要形成更完整的性能结论，仍建议补充 `Infer Every N Frames=1/3` 等跳帧参数对照组。

### 4. 部署包当前能力

部署包位于：

```text
D:/FAS_Train/deploy/FAS_Deploy_Minimal_20260415_170351/
```

核心文件：

- `deploy_and_run.bat`：单脚本启动入口
- `env_name.txt`：配置目标 Miniconda 环境名
- `ui/app.py`：GUI 主程序
- `infer_onnx.py`：单图推理
- `infer_onnx_camera.py`：摄像头推理
- `summarize_gui_runs.py`：GUI 实验日志汇总
- `analyze_threshold.py`：根据带标签 GUI 日志扫描最佳阈值
- `collect_camera_samples.py`：采集真实摄像头微调样本
- `weights/best_model.onnx`：部署模型
- `weights/best_model_me.onnx`：个人真实环境微调后的部署模型，GUI 默认优先加载
- `README_部署说明.md`：部署说明

当前启动逻辑：

```text
查找 conda.bat -> 读取 env_name.txt -> 激活 conda 环境 -> python ui/app.py
```

部署端原则上不需要安装 `torch`、`torchvision`、`albumentations` 或 `onnx`，只需支持 ONNX Runtime、OpenCV、NumPy 与 PyQt5。

## 三、项目目录结构

```text
D:/FAS_Train
├── core/
│   ├── dataset.py
│   ├── model.py
│   └── loss.py
├── data/
│   └── CASIA-SURF/
├── utils/
│   ├── logger.py
│   ├── plot.py
│   └── __init__.py
├── ui/
│   └── app.py
├── weights/
│   ├── best_model.pth
│   ├── last_model.pth
│   ├── best_model.onnx
│   ├── best_model_me.pth
│   ├── best_model_me.onnx
│   ├── training_log.csv
│   ├── training_log.txt
│   ├── split_manifest.json
│   ├── loss_curve.png
│   ├── branch_loss_curve.png
│   ├── accuracy_curve.png
│   ├── pad_metrics_curve.png
│   ├── infer_outputs/
│   └── gui_runs/
├── deploy/
│   ├── FAS_Deploy_Minimal_20260415_170351/
│   └── FAS_Deploy_Minimal_20260415_170351.zip
├── thesis/
├── train.py
├── finetune_me.py
├── export.py
├── infer_onnx.py
├── infer_onnx_batch.py
├── infer_onnx_camera.py
├── summarize_gui_runs.py
├── minimal_validation.py
└── FAS_Project_Context.md
```

## 四、模型与训练方案

### 1. 数据集与监督方式

- 使用数据集：`CASIA-SURF`
- 当前训练列表：`D:/FAS_Train/data/CASIA-SURF/train/train_list.txt`
- 当前公开验证列表：`D:/FAS_Train/data/CASIA-SURF/valid/val_public_list.txt`
- 注意：`val_public_list.txt` 当前不含标签，不能直接计算 Accuracy、APCER、BPCER、ACER。

### 2. 数据划分

- 总样本数：`29266`
- 攻击样本数：`20324`
- 真人样本数：`8942`
- 受试者总数：`300`
- 划分方式：subject-level train/val 划分
- 训练样本数：`23514`
- 验证样本数：`5752`
- 固定随机种子：`42`

### 3. 模型结构

- Backbone：`MobileNetV3-Small`
- 输入：`RGB [B, 3, 224, 224]`
- 输出：
  - 分类分支：`cls_logits [B, 2]`
  - 深度分支：`depth_pred [B, 1, 32, 32]`

### 4. 损失函数

- 分类损失：`CrossEntropyLoss`
- 深度损失：`MSELoss`
- 总损失：`Total Loss = cls_weight * CE + depth_weight * MSE`

### 5. 训练策略

- 使用 subject-level 划分，降低按帧随机划分造成的数据泄漏风险
- 验证指标：Accuracy、APCER、BPCER、ACER
- 最优权重保存标准：验证集 `ACER` 最小
- 支持从 `weights/last_model.pth` 断点续训

## 五、当前实验结果

当前 baseline 已完成一次正式训练，共 `50` 个 epoch。

关键产物：

- `weights/best_model.pth`
- `weights/last_model.pth`
- `weights/best_model.onnx`
- `weights/training_log.csv`
- `weights/training_log.txt`
- `weights/split_manifest.json`
- `weights/loss_curve.png`
- `weights/branch_loss_curve.png`
- `weights/accuracy_curve.png`
- `weights/pad_metrics_curve.png`

当前内部 subject-level 划分下的最优结果：

- 总训练轮数：`50`
- 最优 epoch：`16`
- `val_acc = 100.00%`
- `APCER = 0.0000`
- `BPCER = 0.0000`
- `ACER = 0.0000`

正确解释：

- 可以证明训练代码、模型结构、损失函数、日志系统和部署导出链路已经打通。
- 可以作为毕业论文第 5 章内部验证结果。
- 不能直接证明跨数据集泛化能力。
- 不能直接等同于公开测试集指标。

### 个人真实环境小样本微调结果

为缓解真实摄像头输入与 CASIA-SURF 训练分布之间的差异，已额外采集个人真实环境样本并完成一次小样本微调：

- 微调数据：`data/me/attack` 50 张，`data/me/real` 50 张
- 划分方式：按类别分层随机划分，训练 80 张，验证 20 张
- 基础权重：`weights/best_model.pth`
- 微调脚本：`finetune_me.py`
- 训练设置：`last_blocks`，混合全图/上下文裁剪/紧脸裁剪，验证使用上下文裁剪，BatchNorm 统计冻结，关闭随机增强，`80` epoch，`lr=2e-4`
- 最优输出目录：`weights/me_finetune_20260430_073741/`
- 最新微调权重副本：`weights/best_model_me.pth`
- 最新微调 ONNX 副本：`weights/best_model_me.onnx`

当前微调验证集最优结果：

- 最优 epoch：`49`
- `val_acc = 85.00%`
- `APCER = 0.2000`
- `BPCER = 0.1000`
- `ACER = 0.1500`

补充说明：

- 新模型的概率标尺与上一版不同，推荐 GUI 阈值为 `0.60`。
- 使用 `Face Crop Margin=0.80` 时会保留更多脸部周围上下文，更适合检测手机/屏幕照片攻击；过小的紧脸裁剪容易丢失攻击线索。

解释边界：

- 该结果可用于说明“通过采集部署环境样本进行小规模适配后，真实环境验证集表现得到改善”。
- 该结果只覆盖当前个人设备、当前光照和当前攻击采集方式，不代表通用跨设备或跨数据集泛化能力。
- 论文中应将其表述为“真实场景适配实验”或“个人环境微调实验”，不要写成公开测试集指标。

## 六、当前推理与部署能力

### 1. 单图推理

- 脚本：`infer_onnx.py`
- 功能：加载 ONNX，对单张图片输出 Attack/Genuine 概率，可保存伪深度图。

### 2. 批量推理

- 脚本：`infer_onnx_batch.py`
- 功能：支持 list 文件或图片目录输入，输出逐样本 CSV；若含标签则统计 Accuracy、APCER、BPCER、ACER。

### 3. 摄像头实时推理

- 脚本：`infer_onnx_camera.py`
- 功能：摄像头实时推理，显示类别、概率、耗时、FPS，可选伪深度图。
- 已新增：`--face_crop` 与 `--crop_margin`，支持 OpenCV 轻量人脸 ROI 裁剪。

### 4. GUI 推理

- 脚本：`ui/app.py`
- 功能：实时 GUI 演示、参数调节、低照度增强、人脸裁剪、日志记录、截图保存、结果目录打开。
- 模型加载策略：默认优先加载 `weights/best_model_me.onnx`，若不存在则回退到 `weights/best_model.onnx`。

### 5. GUI 日志汇总

- 脚本：`summarize_gui_runs.py`
- 输入：`weights/gui_runs/run_*/session_log.csv`
- 输出：
  - `weights/gui_runs/gui_runs_summary.csv`
  - `weights/gui_runs/gui_runs_summary.md`
- 用途：整理论文第 5 章端侧 FPS、推理耗时、亮度、人脸检出率等测试表。

### 6. 阈值分析与真实样本采集

- 脚本：`analyze_threshold.py`
- 输入：带 `true_label` 的 GUI `session_log.csv`
- 输出：`weights/gui_runs/threshold_analysis.csv`
- 用途：根据真实摄像头日志自动扫描 Genuine 阈值，统计 Accuracy、APCER、BPCER、ACER。

当前带标签日志分析结果：

- 带标签样本数：`339`
- 攻击样本数：`122`
- 真人样本数：`217`
- 攻击样本平均 `Genuine prob`：`0.4794`
- 真人样本平均 `Genuine prob`：`0.3026`
- 当前扫描到的最优阈值：`0.83`
- 最优阈值下 `Accuracy=0.3864`，`APCER=0.0000`，`BPCER=0.9585`，`ACER=0.4793`

结论：当前真实摄像头场景下，单纯调阈值无法解决识别效果问题，需要采集真实摄像头样本并进行小规模微调。

- 脚本：`collect_camera_samples.py`
- 输出：`data/real_camera_finetune/`
- 用途：采集真人与攻击 RGB 样本，用于后续真实场景微调。

### 7. 个人真实环境微调

- 脚本：`finetune_me.py`
- 输入：`data/me/attack` 与 `data/me/real`
- 输出：`weights/me_finetune_*/`、`weights/best_model_me.pth`、`weights/best_model_me.onnx`
- 默认策略：加载 baseline 权重，冻结大部分 backbone，仅微调末端特征层和分类头；默认启用人脸裁剪，保存 CSV 日志、split manifest、checkpoint，并导出 ONNX。
- 推荐命令：

```text
conda run -n FAS_Train python D:/FAS_Train/finetune_me.py --data_dir D:/FAS_Train/data/me --checkpoint D:/FAS_Train/weights/best_model.pth --epochs 80 --batch_size 8 --lr 0.0002 --train_mode last_blocks --val_ratio 0.2 --seed 42 --face_margin 1.35 --no_augment
```

## 七、论文写作边界

论文可以写：

- MobileNetV3-Small 双分支人脸防伪模型
- 伪深度监督
- subject-level 划分
- ONNX 导出与 ONNX Runtime 推理
- 单图、批量、摄像头、GUI 推理
- OpenCV Haar cascade 轻量人脸 ROI 裁剪
- 低照度增强
- GUI 实验日志记录与汇总
- 带标签日志阈值分析
- 真实摄像头微调样本采集
- 个人真实环境小样本微调
- 最小部署包与 Miniconda 单脚本启动

论文不能写成已完成：

- YOLO 或 RetinaFace 人脸检测
- 注意力机制及其消融实验
- 域泛化模块或跨数据集泛化实验
- 公开测试集正式指标
- 在任意配置下稳定达到 `15 FPS` 或更高；目前只能写已有 GUI 日志对应配置下的阶段性结果
- 3D 面具等完整黑盒攻击验证
- 手机 App、Web 服务或云端系统

## 八、仍需外部条件完成的项目

以下项目不是当前代码仓库单独能完成的，必须依赖外部数据、硬件或正式测试过程：

1. 笔记本正式 FPS 测试：
   - 已完成 3 组 USB 手机摄像头测试
   - 建议继续测试不同分辨率、跳帧参数、低照度开关、人脸裁剪开关
   - 每次测试完成后运行 `python summarize_gui_runs.py`

2. 正式运行截图：
   - 已保存一组截图和伪深度图
   - 建议继续保存真人、照片/屏幕攻击、暗光、低照度增强、人脸裁剪框截图

3. 公开测试集指标：
   - 需要带标签的公开验证/测试列表或官方评测结果
   - 当前 `val_public_list.txt` 不含标签，不能计算正式指标

4. 跨数据集泛化：
   - 需要额外有标签的人脸防伪数据集
   - 需要新增 Dataset 适配、测试脚本和实验表

5. 注意力机制消融实验：
   - 需要修改 `core/model.py`
   - 需要重新训练 baseline 与注意力版本
   - 需要消融结果表支撑

## 九、后续最小工作建议

如果目标是尽快完成毕业论文和答辩材料，优先顺序如下：

1. 在笔记本运行最新部署包，收集 GUI 日志和截图。
2. 使用 `weights/best_model_me.onnx` 在笔记本上重新做真人/攻击对照测试。
3. 运行 `python summarize_gui_runs.py` 生成 FPS/耗时汇总表。
4. 将 `training_log.csv`、训练曲线、GUI 截图、GUI 汇总表和微调结果填入论文第 5 章。
5. 清理 Word 初稿模板提示文字，按 `thesis/毕业论文大纲与写作建议.md` 调整章节。
6. 最后写摘要、英文摘要、结论和致谢。

## 十、给后续 AI 协作的明确结论

如果后续 AI 再接手本仓库，请默认：

- 项目主线代码已经基本完成，不是从零开发阶段。
- 最重要的剩余工作是正式笔记本测试证据、论文正文整合和截图材料。
- 当前 GUI 会优先加载 `weights/best_model_me.onnx`，这是个人真实环境微调模型；原始 baseline 仍为 `weights/best_model.onnx`。
- 所有论文结论必须有真实日志、截图、CSV、曲线或权重支撑。
- 不允许把展望内容写成已完成内容。
