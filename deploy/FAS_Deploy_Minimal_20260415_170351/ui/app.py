# -*- coding: utf-8 -*-
"""
ui/app.py - 基于 PyQt5 的人脸防伪桌面演示系统

当前版本重点增强：
    1. 自动记录实验数据（CSV）
    2. 支持手动保存截图
    3. 支持低照度增强开关
    4. 支持降低采集分辨率与跳帧推理以提升实时性
"""

import csv
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QComboBox,
    QSpinBox,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from infer_onnx import PROJECT_ROOT as APP_ROOT, LABEL_NAMES, build_session, run_inference, softmax
from infer_onnx_camera import (
    open_camera,
    preprocess_bgr_frame,
    make_depth_vis,
    create_face_detector,
    detect_faces,
    detect_largest_face,
    crop_face_region,
)


WINDOW_TITLE = 'FAS Face Anti-Spoofing System'
LOW_BRIGHTNESS_THRESHOLD = 70.0
PROB_SMOOTH_ALPHA = 0.35
DECISION_HYSTERESIS = 0.08
SWITCH_CONFIRM_FRAMES = 3


def resolve_default_model_path() -> str:
    adapted_model = os.path.join(APP_ROOT, 'weights', 'best_model_me.onnx')
    baseline_model = os.path.join(APP_ROOT, 'weights', 'best_model.onnx')
    return adapted_model if os.path.exists(adapted_model) else baseline_model


def cv2_bgr_to_qpixmap(frame_bgr, target_size=None) -> QPixmap:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, c = rgb.shape
    bytes_per_line = w * c
    q_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()

    pixmap = QPixmap.fromImage(q_image)
    if target_size is not None and not target_size.isEmpty():
        pixmap = pixmap.scaled(
            target_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
    return pixmap


def enhance_low_light(frame_bgr: np.ndarray) -> np.ndarray:
    """使用 CLAHE + 轻量 gamma 增强，改善室内偏暗场景。"""
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)
    enhanced = cv2.merge((l_enhanced, a_channel, b_channel))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    gamma = 1.15
    lut = np.array([
        np.clip(((i / 255.0) ** (1.0 / gamma)) * 255.0, 0, 255)
        for i in range(256)
    ]).astype(np.uint8)
    return cv2.LUT(enhanced, lut)


class FASMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.model_path = resolve_default_model_path()
        self.output_root = os.path.join(APP_ROOT, 'weights', 'gui_runs')
        os.makedirs(self.output_root, exist_ok=True)

        self.session = None
        self.capture = None
        self.face_detector = create_face_detector()
        self.is_detecting = False
        self.display_fps = 0.0
        self.infer_fps = 0.0
        self.last_display_t = time.perf_counter()

        self.frame_index = 0
        self.latest_result = None
        self.current_frame_bgr = None
        self.current_depth_bgr = None
        self.latest_face_box = None
        self.smoothed_face_box = None
        self.face_missing_count = 0
        self.genuine_prob_ema = None
        self.stable_pred_label = None
        self.pending_pred_label = None
        self.pending_pred_count = 0

        self.run_dir = ''
        self.log_csv_path = ''
        self.is_logging_active = False

        self.timer = QTimer(self)
        self.timer.setInterval(30)
        self.timer.timeout.connect(self.update_frame)

        self.setWindowTitle(WINDOW_TITLE)
        self.resize(1380, 860)

        self.build_ui()
        self.refresh_model_ui()
        self.reset_detection_info()
        self.lbl_camera_status.setText('摄像头状态: 未开启')
        self.lbl_camera_status.setStyleSheet('color: #444;')

    def build_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)

        root_layout = QVBoxLayout(central)

        title = QLabel('面向身份认证的人脸呈现攻击检测系统')
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont('Microsoft YaHei UI', 18, QFont.Bold))
        root_layout.addWidget(title)

        subtitle = QLabel('ONNX Runtime + OpenCV + PyQt5')
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet('color: #666;')
        root_layout.addWidget(subtitle)

        content_layout = QHBoxLayout()
        root_layout.addLayout(content_layout)

        content_layout.addWidget(self.build_display_panel(), stretch=3)
        content_layout.addWidget(self.build_control_panel(), stretch=2)

        status_bar = QStatusBar(self)
        self.setStatusBar(status_bar)
        self.statusBar().showMessage('系统就绪，请先加载模型并打开摄像头。')

    def build_display_panel(self):
        panel = QGroupBox('实时画面')
        layout = QGridLayout(panel)

        self.camera_label = QLabel('摄像头画面')
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(800, 520)
        self.camera_label.setStyleSheet(
            'background-color: #111; color: #ddd; border: 1px solid #444;'
        )
        layout.addWidget(self.camera_label, 0, 0)

        self.depth_label = QLabel('伪深度图')
        self.depth_label.setAlignment(Qt.AlignCenter)
        self.depth_label.setMinimumSize(320, 240)
        self.depth_label.setStyleSheet(
            'background-color: #1b1b1b; color: #ddd; border: 1px solid #444;'
        )
        layout.addWidget(self.depth_label, 1, 0)

        return panel

    def build_control_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)

        layout.addWidget(self.build_model_group())
        layout.addWidget(self.build_camera_group())
        layout.addWidget(self.build_action_group())
        layout.addWidget(self.build_info_group())
        layout.addStretch()
        return panel

    def build_model_group(self):
        group = QGroupBox('模型设置')
        layout = QVBoxLayout(group)

        self.model_path_edit = QLineEdit()
        self.model_path_edit.setReadOnly(True)
        layout.addWidget(self.model_path_edit)

        row = QHBoxLayout()
        self.btn_choose_model = QPushButton('选择 ONNX')
        self.btn_choose_model.clicked.connect(self.choose_model_file)
        row.addWidget(self.btn_choose_model)

        self.btn_load_model = QPushButton('加载模型')
        self.btn_load_model.clicked.connect(self.load_model)
        row.addWidget(self.btn_load_model)
        layout.addLayout(row)
        return group

    def build_camera_group(self):
        group = QGroupBox('检测与性能设置')
        layout = QFormLayout(group)

        self.camera_id_spin = QSpinBox()
        self.camera_id_spin.setRange(0, 9)
        self.camera_id_spin.setValue(0)
        layout.addRow('Camera ID', self.camera_id_spin)

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setValue(0.60)
        layout.addRow('Threshold', self.threshold_spin)

        self.width_spin = QSpinBox()
        self.width_spin.setRange(320, 1920)
        self.width_spin.setSingleStep(160)
        self.width_spin.setValue(640)
        layout.addRow('Capture Width', self.width_spin)

        self.height_spin = QSpinBox()
        self.height_spin.setRange(240, 1080)
        self.height_spin.setSingleStep(120)
        self.height_spin.setValue(480)
        layout.addRow('Capture Height', self.height_spin)

        self.infer_every_spin = QSpinBox()
        self.infer_every_spin.setRange(1, 10)
        self.infer_every_spin.setValue(2)
        layout.addRow('Infer Every N Frames', self.infer_every_spin)

        self.flip_check = QCheckBox('水平镜像显示')
        self.flip_check.setChecked(True)
        layout.addRow(self.flip_check)

        self.low_light_check = QCheckBox('启用低照度增强')
        self.low_light_check.setChecked(False)
        layout.addRow(self.low_light_check)

        self.face_crop_check = QCheckBox('启用人脸裁剪')
        self.face_crop_check.setChecked(True)
        layout.addRow(self.face_crop_check)

        self.crop_margin_spin = QDoubleSpinBox()
        self.crop_margin_spin.setRange(0.0, 1.5)
        self.crop_margin_spin.setSingleStep(0.05)
        self.crop_margin_spin.setValue(0.80)
        layout.addRow('Face Crop Margin', self.crop_margin_spin)

        self.label_combo = QComboBox()
        self.label_combo.addItem('未知 / 不计入阈值分析', '')
        self.label_combo.addItem('真人 Genuine', '1')
        self.label_combo.addItem('攻击 Attack', '0')
        layout.addRow('Sample Label', self.label_combo)

        self.record_check = QCheckBox('记录实验数据 (CSV)')
        self.record_check.setChecked(True)
        layout.addRow(self.record_check)

        return group

    def build_action_group(self):
        group = QGroupBox('操作面板')
        layout = QVBoxLayout(group)

        self.btn_open_camera = QPushButton('打开摄像头')
        self.btn_open_camera.clicked.connect(self.open_camera_clicked)
        layout.addWidget(self.btn_open_camera)

        self.btn_close_camera = QPushButton('关闭摄像头')
        self.btn_close_camera.clicked.connect(self.close_camera)
        self.btn_close_camera.setEnabled(False)
        layout.addWidget(self.btn_close_camera)

        self.btn_start_detect = QPushButton('开始检测')
        self.btn_start_detect.clicked.connect(self.start_detection)
        self.btn_start_detect.setEnabled(False)
        layout.addWidget(self.btn_start_detect)

        self.btn_stop_detect = QPushButton('停止检测')
        self.btn_stop_detect.clicked.connect(self.stop_detection)
        self.btn_stop_detect.setEnabled(False)
        layout.addWidget(self.btn_stop_detect)

        self.btn_save_snapshot = QPushButton('保存截图')
        self.btn_save_snapshot.clicked.connect(self.save_snapshot)
        self.btn_save_snapshot.setEnabled(False)
        layout.addWidget(self.btn_save_snapshot)

        self.btn_open_output = QPushButton('打开结果目录')
        self.btn_open_output.clicked.connect(self.open_output_dir)
        layout.addWidget(self.btn_open_output)

        self.btn_exit = QPushButton('退出程序')
        self.btn_exit.clicked.connect(self.close)
        layout.addWidget(self.btn_exit)

        return group

    def build_info_group(self):
        group = QGroupBox('状态信息')
        layout = QVBoxLayout(group)

        mono_font = QFont('Consolas', 10)

        self.lbl_model_status = QLabel()
        self.lbl_model_status.setFont(mono_font)
        self.lbl_model_status.setWordWrap(True)
        layout.addWidget(self.lbl_model_status)

        self.lbl_camera_status = QLabel()
        self.lbl_camera_status.setFont(mono_font)
        layout.addWidget(self.lbl_camera_status)

        self.lbl_prediction = QLabel()
        self.lbl_prediction.setFont(QFont('Consolas', 12, QFont.Bold))
        layout.addWidget(self.lbl_prediction)

        self.lbl_attack_prob = QLabel()
        self.lbl_attack_prob.setFont(mono_font)
        layout.addWidget(self.lbl_attack_prob)

        self.lbl_genuine_prob = QLabel()
        self.lbl_genuine_prob.setFont(mono_font)
        layout.addWidget(self.lbl_genuine_prob)

        self.lbl_infer_time = QLabel()
        self.lbl_infer_time.setFont(mono_font)
        layout.addWidget(self.lbl_infer_time)

        self.lbl_fps = QLabel()
        self.lbl_fps.setFont(mono_font)
        layout.addWidget(self.lbl_fps)

        self.lbl_brightness = QLabel()
        self.lbl_brightness.setFont(mono_font)
        layout.addWidget(self.lbl_brightness)

        self.lbl_record_status = QLabel()
        self.lbl_record_status.setFont(mono_font)
        self.lbl_record_status.setWordWrap(True)
        layout.addWidget(self.lbl_record_status)

        return group

    def refresh_model_ui(self):
        self.model_path_edit.setText(self.model_path)
        if self.session is None:
            self.lbl_model_status.setText(f'模型状态: 未加载\n模型路径: {self.model_path}')
            self.lbl_model_status.setStyleSheet('color: #444;')
        else:
            self.lbl_model_status.setText(f'模型状态: 已加载\n模型路径: {self.model_path}')
            self.lbl_model_status.setStyleSheet('color: green;')

    def choose_model_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            '选择 ONNX 模型',
            os.path.join(APP_ROOT, 'weights'),
            'ONNX Files (*.onnx)',
        )
        if file_path:
            self.model_path = file_path
            self.session = None
            self.refresh_model_ui()
            self.set_status('已选择模型文件，请点击“加载模型”。')

    def load_model(self):
        if not os.path.exists(self.model_path):
            QMessageBox.critical(self, '错误', f'找不到模型文件：\n{self.model_path}')
            self.refresh_model_ui()
            return

        try:
            self.session = build_session(self.model_path)
            self.refresh_model_ui()
            self.set_status('ONNX 模型加载成功。')
            QMessageBox.information(self, '成功', 'ONNX 模型加载成功。')
            if self.capture is not None and self.capture.isOpened():
                self.btn_start_detect.setEnabled(True)
        except Exception as exc:
            self.session = None
            self.refresh_model_ui()
            QMessageBox.critical(self, '错误', f'加载模型失败：\n{exc}')
            self.set_status('模型加载失败。')

    def open_camera_clicked(self):
        if self.capture is not None and self.capture.isOpened():
            return

        try:
            self.capture = open_camera(
                self.camera_id_spin.value(),
                width=self.width_spin.value(),
                height=self.height_spin.value(),
            )
        except Exception as exc:
            self.capture = None
            self.lbl_camera_status.setText(f'摄像头状态: 打开失败 ({exc})')
            self.lbl_camera_status.setStyleSheet('color: red;')
            QMessageBox.warning(self, '摄像头错误', str(exc))
            self.set_status('摄像头打开失败。')
            return

        self.lbl_camera_status.setText(
            f'摄像头状态: 已开启 (ID={self.camera_id_spin.value()}, '
            f'{self.width_spin.value()}x{self.height_spin.value()})'
        )
        self.lbl_camera_status.setStyleSheet('color: green;')
        self.btn_open_camera.setEnabled(False)
        self.btn_close_camera.setEnabled(True)
        self.btn_start_detect.setEnabled(self.session is not None)
        self.btn_save_snapshot.setEnabled(True)
        self.display_fps = 0.0
        self.infer_fps = 0.0
        self.last_display_t = time.perf_counter()
        self.frame_index = 0
        self.timer.start()
        self.set_status('摄像头已开启。')

    def close_camera(self):
        self.stop_detection(reset_camera_only=True)
        self.timer.stop()

        if self.capture is not None:
            self.capture.release()
            self.capture = None

        self.btn_open_camera.setEnabled(True)
        self.btn_close_camera.setEnabled(False)
        self.btn_start_detect.setEnabled(False)
        self.btn_stop_detect.setEnabled(False)
        self.btn_save_snapshot.setEnabled(False)

        self.camera_label.setText('摄像头画面')
        self.camera_label.setPixmap(QPixmap())
        self.depth_label.setText('伪深度图')
        self.depth_label.setPixmap(QPixmap())
        self.lbl_camera_status.setText('摄像头状态: 未开启')
        self.lbl_camera_status.setStyleSheet('color: #444;')
        self.current_frame_bgr = None
        self.current_depth_bgr = None
        self.latest_result = None
        self.latest_face_box = None
        self.smoothed_face_box = None
        self.face_missing_count = 0
        self.set_status('摄像头已关闭。')

    def start_detection(self):
        if self.session is None:
            QMessageBox.warning(self, '提示', '请先加载 ONNX 模型。')
            return
        if self.capture is None or not self.capture.isOpened():
            QMessageBox.warning(self, '提示', '请先打开摄像头。')
            return

        self.is_detecting = True
        self.display_fps = 0.0
        self.infer_fps = 0.0
        self.last_display_t = time.perf_counter()
        self.frame_index = 0
        self.latest_result = None
        self.reset_temporal_state()
        self.prepare_run_dir()
        self.btn_start_detect.setEnabled(False)
        self.btn_stop_detect.setEnabled(True)
        self.set_status('实时检测已开始。')

    def stop_detection(self, reset_camera_only=False):
        self.is_detecting = False
        self.is_logging_active = False
        self.reset_temporal_state()
        if self.capture is not None and self.capture.isOpened():
            self.btn_start_detect.setEnabled(self.session is not None and not reset_camera_only)
        else:
            self.btn_start_detect.setEnabled(False)
        self.btn_stop_detect.setEnabled(False)
        self.reset_detection_info()
        if not reset_camera_only:
            self.set_status('实时检测已停止。')

    def reset_detection_info(self):
        self.lbl_prediction.setText('预测结果: -')
        self.lbl_prediction.setStyleSheet('color: #1f5aa6;')
        self.lbl_attack_prob.setText('Attack prob: -')
        self.lbl_genuine_prob.setText('Genuine prob: -')
        self.lbl_infer_time.setText('推理耗时: - ms')
        self.lbl_fps.setText('Display FPS: - | Infer FPS: -')
        self.lbl_brightness.setText('Brightness: -')
        if self.record_check.isChecked():
            text = self.run_dir if self.run_dir else '待开始检测后生成'
            self.lbl_record_status.setText(f'实验记录目录: {text}')
        else:
            self.lbl_record_status.setText('实验记录目录: 已关闭自动记录')
        self.depth_label.setText('伪深度图')
        self.depth_label.setPixmap(QPixmap())

    def prepare_run_dir(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = os.path.join(self.output_root, f'run_{timestamp}')
        os.makedirs(self.run_dir, exist_ok=True)
        self.log_csv_path = os.path.join(self.run_dir, 'session_log.csv')
        self.is_logging_active = self.record_check.isChecked()

        if self.is_logging_active:
            with open(self.log_csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'frame_index', 'pred_label', 'pred_name',
                    'attack_prob', 'genuine_prob', 'threshold',
                    'infer_every_n_frames',
                    'inference_ms', 'display_fps', 'infer_fps',
                    'brightness_mean', 'low_light_enabled',
                    'face_crop_enabled', 'face_detected', 'face_box', 'crop_margin',
                    'true_label', 'true_name',
                    'camera_id', 'capture_width', 'capture_height',
                    'model_path',
                    'raw_attack_prob', 'raw_genuine_prob', 'stability_state'
                ])

        self.lbl_record_status.setText(
            f'实验记录目录: {self.run_dir}' if self.record_check.isChecked()
            else f'实验记录目录: {self.run_dir} (仅手动截图)'
        )

    def append_log_row(self, result):
        if not self.is_logging_active or not self.log_csv_path:
            return

        with open(self.log_csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            pred_label = result.get('pred_label', -1)
            pred_name = LABEL_NAMES.get(pred_label, result.get('pred_name', 'No Face'))
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                self.frame_index,
                pred_label,
                pred_name,
                self.format_optional_float(result.get('attack_prob')),
                self.format_optional_float(result.get('genuine_prob')),
                f"{self.threshold_spin.value():.2f}",
                self.infer_every_spin.value(),
                self.format_optional_float(result.get('inference_ms'), digits=2),
                f"{self.display_fps:.2f}",
                f"{self.infer_fps:.2f}",
                self.format_optional_float(result.get('brightness_mean'), digits=2),
                int(self.low_light_check.isChecked()),
                int(self.face_crop_check.isChecked()),
                int(result.get('face_detected', False)),
                result.get('face_box', ''),
                f"{self.crop_margin_spin.value():.2f}",
                self.label_combo.currentData(),
                self.label_combo.currentText(),
                self.camera_id_spin.value(),
                self.width_spin.value(),
                self.height_spin.value(),
                self.model_path,
                self.format_optional_float(result.get('raw_attack_prob')),
                self.format_optional_float(result.get('raw_genuine_prob')),
                result.get('stability_state', ''),
            ])

    @staticmethod
    def format_optional_float(value, digits=6):
        if value is None:
            return ''
        try:
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return ''

    def save_snapshot(self):
        if self.current_frame_bgr is None:
            QMessageBox.information(self, '提示', '当前还没有可保存的画面。')
            return

        if not self.run_dir:
            self.prepare_run_dir()

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        frame_path = os.path.join(self.run_dir, f'frame_{timestamp}.jpg')
        cv2.imwrite(frame_path, self.current_frame_bgr)

        depth_path = ''
        if self.current_depth_bgr is not None:
            depth_path = os.path.join(self.run_dir, f'depth_{timestamp}.png')
            cv2.imwrite(depth_path, self.current_depth_bgr)

        msg = f'截图已保存：\n{frame_path}'
        if depth_path:
            msg += f'\n{depth_path}'
        QMessageBox.information(self, '保存成功', msg)
        self.set_status('当前帧截图已保存。')

    def open_output_dir(self):
        target = self.run_dir if self.run_dir else self.output_root
        os.makedirs(target, exist_ok=True)
        try:
            os.startfile(target)  # type: ignore[attr-defined]
        except Exception as exc:
            QMessageBox.warning(self, '提示', f'无法打开目录：\n{target}\n\n{exc}')

    def update_frame(self):
        if self.capture is None or not self.capture.isOpened():
            return

        ok, frame = self.capture.read()
        if not ok or frame is None:
            self.set_status('读取摄像头画面失败。')
            return

        if self.flip_check.isChecked():
            frame = cv2.flip(frame, 1)

        now = time.perf_counter()
        loop_fps = 1.0 / max(now - self.last_display_t, 1e-6)
        self.display_fps = loop_fps if self.display_fps == 0.0 else 0.9 * self.display_fps + 0.1 * loop_fps
        self.last_display_t = now

        display = frame.copy()
        self.frame_index += 1

        if self.is_detecting and self.session is not None:
            infer_every = max(1, self.infer_every_spin.value())
            need_infer = self.latest_result is None or (self.frame_index % infer_every == 0)

            if need_infer:
                try:
                    infer_frame = enhance_low_light(frame) if self.low_light_check.isChecked() else frame
                    face_box = None
                    face_detected = False
                    if self.face_crop_check.isChecked():
                        detected_faces = detect_faces(infer_frame, self.face_detector)
                        if len(detected_faces) == 0:
                            self.clear_face_tracking()
                            result = self.make_no_face_result(infer_frame)
                            self.latest_result = result
                            self.append_log_row(result)
                        elif len(detected_faces) > 1:
                            self.clear_face_tracking()
                            result = self.make_invalid_face_result(infer_frame, len(detected_faces))
                            self.latest_result = result
                            self.append_log_row(result)
                        else:
                            detected_box = detected_faces[0]
                            face_box = self.update_face_box(detected_box, infer_frame.shape)
                            infer_frame, crop_box = crop_face_region(
                                infer_frame,
                                face_box,
                                margin_ratio=self.crop_margin_spin.value(),
                            )
                            face_detected = True
                            face_box = crop_box
                            result = self.infer_on_frame(infer_frame)
                            result['face_detected'] = face_detected
                            result['face_box'] = self.format_face_box(face_box)
                            self.latest_face_box = face_box
                            self.latest_result = result
                            self.append_log_row(result)
                    else:
                        result = self.infer_on_frame(infer_frame)
                        result['face_detected'] = face_detected
                        result['face_box'] = self.format_face_box(face_box)
                        self.latest_face_box = face_box
                        self.latest_result = result
                        self.append_log_row(result)
                except Exception as exc:
                    self.is_detecting = False
                    self.btn_start_detect.setEnabled(True)
                    self.btn_stop_detect.setEnabled(False)
                    QMessageBox.warning(self, '推理错误', f'实时推理失败：\n{exc}')
                    self.set_status('实时推理失败，已自动停止。')

            if self.latest_result is not None:
                self.apply_inference_result(display, self.latest_result)

        self.current_frame_bgr = display.copy()
        self.show_frame_on_label(display, self.camera_label)

    def clear_face_tracking(self):
        self.latest_face_box = None
        self.smoothed_face_box = None
        self.face_missing_count = 0
        self.reset_temporal_state()

    def reset_temporal_state(self):
        self.genuine_prob_ema = None
        self.stable_pred_label = None
        self.pending_pred_label = None
        self.pending_pred_count = 0

    def make_no_face_result(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return {
            'pred_label': -1,
            'pred_name': 'No Face',
            'attack_prob': None,
            'genuine_prob': None,
            'inference_ms': None,
            'brightness_mean': float(gray.mean()),
            'depth_pred': None,
            'face_detected': False,
            'face_box': '',
        }

    def make_invalid_face_result(self, frame_bgr, face_count):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return {
            'pred_label': -2,
            'pred_name': 'Invalid',
            'message': f'检测到 {face_count} 张人脸，请只保留一张待验证人脸',
            'attack_prob': None,
            'genuine_prob': None,
            'inference_ms': None,
            'brightness_mean': float(gray.mean()),
            'depth_pred': None,
            'face_detected': True,
            'face_box': '',
        }

    def update_face_box(self, detected_box, frame_shape):
        if detected_box is None:
            self.face_missing_count += 1
            if self.face_missing_count <= 8 and self.smoothed_face_box is not None:
                return self.smoothed_face_box
            self.smoothed_face_box = None
            return None

        self.face_missing_count = 0
        box = np.array(detected_box, dtype=np.float32)
        if self.smoothed_face_box is None:
            smoothed = box
        else:
            previous = np.array(self.smoothed_face_box, dtype=np.float32)
            smoothed = 0.65 * previous + 0.35 * box

        x, y, w, h = [int(round(v)) for v in smoothed]
        height, width = frame_shape[:2]
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = max(1, min(w, width - x))
        h = max(1, min(h, height - y))
        self.smoothed_face_box = (x, y, w, h)
        return self.smoothed_face_box

    def format_face_box(self, face_box):
        if face_box is None:
            return ''
        x, y, w, h = [int(v) for v in face_box]
        return f'{x},{y},{w},{h}'

    def infer_on_frame(self, frame_bgr):
        input_tensor = preprocess_bgr_frame(frame_bgr)

        infer_t0 = time.perf_counter()
        cls_logits, depth_pred = run_inference(self.session, input_tensor)
        inference_ms = (time.perf_counter() - infer_t0) * 1000.0

        probs = softmax(cls_logits)[0]
        raw_attack_prob = float(probs[0])
        raw_genuine_prob = float(probs[1])
        threshold = float(self.threshold_spin.value())
        pred_label, attack_prob, genuine_prob, stability_state = self.stabilize_prediction(
            raw_attack_prob,
            raw_genuine_prob,
            threshold,
        )

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        brightness_mean = float(gray.mean())
        infer_fps = 1000.0 / max(inference_ms, 1e-6)

        self.infer_fps = infer_fps if self.infer_fps == 0.0 else 0.9 * self.infer_fps + 0.1 * infer_fps

        return {
            'pred_label': pred_label,
            'attack_prob': attack_prob,
            'genuine_prob': genuine_prob,
            'raw_attack_prob': raw_attack_prob,
            'raw_genuine_prob': raw_genuine_prob,
            'stability_state': stability_state,
            'inference_ms': inference_ms,
            'brightness_mean': brightness_mean,
            'depth_pred': depth_pred,
        }

    def stabilize_prediction(self, raw_attack_prob, raw_genuine_prob, threshold):
        if self.genuine_prob_ema is None:
            self.genuine_prob_ema = raw_genuine_prob
        else:
            self.genuine_prob_ema = (
                (1.0 - PROB_SMOOTH_ALPHA) * self.genuine_prob_ema
                + PROB_SMOOTH_ALPHA * raw_genuine_prob
            )

        smooth_genuine = float(self.genuine_prob_ema)
        smooth_attack = 1.0 - smooth_genuine

        if self.stable_pred_label is None:
            self.stable_pred_label = 1 if smooth_genuine >= threshold else 0
            self.pending_pred_label = None
            self.pending_pred_count = 0
            return self.stable_pred_label, smooth_attack, smooth_genuine, 'init'

        high = min(1.0, threshold + DECISION_HYSTERESIS)
        low = max(0.0, threshold - DECISION_HYSTERESIS)

        candidate = self.stable_pred_label
        if self.stable_pred_label == 1 and smooth_genuine <= low:
            candidate = 0
        elif self.stable_pred_label == 0 and smooth_genuine >= high:
            candidate = 1

        if candidate != self.stable_pred_label:
            if self.pending_pred_label == candidate:
                self.pending_pred_count += 1
            else:
                self.pending_pred_label = candidate
                self.pending_pred_count = 1

            if self.pending_pred_count >= SWITCH_CONFIRM_FRAMES:
                self.stable_pred_label = candidate
                self.pending_pred_label = None
                self.pending_pred_count = 0
                state = 'switched'
            else:
                state = f'pending_{LABEL_NAMES[candidate]}_{self.pending_pred_count}/{SWITCH_CONFIRM_FRAMES}'
        else:
            self.pending_pred_label = None
            self.pending_pred_count = 0
            state = 'stable'

        return self.stable_pred_label, smooth_attack, smooth_genuine, state

    def apply_inference_result(self, display_frame, result):
        pred_label = result['pred_label']
        if pred_label < 0:
            self.apply_invalid_or_no_face_result(display_frame, result)
            return

        pred_name = LABEL_NAMES[pred_label]
        is_genuine = pred_label == 1
        color = (60, 180, 75) if is_genuine else (40, 40, 230)

        self.lbl_prediction.setText(f'预测结果: {pred_name}')
        self.lbl_prediction.setStyleSheet(f'color: {"green" if is_genuine else "red"};')
        self.lbl_attack_prob.setText(f"Attack prob: {result['attack_prob']:.4f}")
        self.lbl_genuine_prob.setText(f"Genuine prob: {result['genuine_prob']:.4f}")
        self.lbl_infer_time.setText(f"推理耗时: {result['inference_ms']:.1f} ms")
        self.lbl_fps.setText(f"Display FPS: {self.display_fps:.1f} | Infer FPS: {self.infer_fps:.1f}")
        brightness_text = (
            f"Brightness: {result['brightness_mean']:.1f} | "
            f"Low-light: {'ON' if self.low_light_check.isChecked() else 'OFF'} | "
            f"Face crop: {'ON' if self.face_crop_check.isChecked() else 'OFF'}"
        )
        if result['brightness_mean'] < LOW_BRIGHTNESS_THRESHOLD:
            self.lbl_brightness.setStyleSheet('color: #c97a00;')
            if not self.low_light_check.isChecked():
                brightness_text += ' | 建议开启低照度增强/补光'
        else:
            self.lbl_brightness.setStyleSheet('color: #222;')
        self.lbl_brightness.setText(brightness_text)

        overlay_text = (
            f"{pred_name} | A:{result['attack_prob']:.3f} "
            f"G:{result['genuine_prob']:.3f} | D:{self.display_fps:.1f} I:{self.infer_fps:.1f}"
        )
        cv2.rectangle(display_frame, (12, 12), (display_frame.shape[1] - 12, display_frame.shape[0] - 12), color, 2)
        if self.face_crop_check.isChecked() and self.latest_face_box is not None:
            x, y, w, h = [int(v) for v in self.latest_face_box]
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 215, 255), 2)
            cv2.putText(
                display_frame,
                'Face ROI',
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 215, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            display_frame,
            overlay_text,
            (24, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            color,
            2,
            cv2.LINE_AA,
        )

        if self.low_light_check.isChecked():
            cv2.putText(
                display_frame,
                'LOW-LIGHT ENHANCEMENT',
                (24, 72),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (0, 215, 255),
                2,
                cv2.LINE_AA,
            )

        if result['brightness_mean'] < LOW_BRIGHTNESS_THRESHOLD:
            cv2.putText(
                display_frame,
                'LOW BRIGHTNESS',
                (24, 104),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (0, 180, 255),
                2,
                cv2.LINE_AA,
            )

        depth_vis = make_depth_vis(result['depth_pred'][0, 0])
        self.current_depth_bgr = depth_vis.copy()
        self.show_frame_on_label(depth_vis, self.depth_label)

    def apply_invalid_or_no_face_result(self, display_frame, result):
        color = (0, 165, 255)
        message = result.get('message', '未检测到人脸')
        overlay = 'No face detected - inference skipped' if result['pred_label'] == -1 else 'Multiple faces detected - verification blocked'
        self.lbl_prediction.setText(f'预测结果: {message}')
        self.lbl_prediction.setStyleSheet('color: #c97a00;')
        self.lbl_attack_prob.setText('Attack prob: -')
        self.lbl_genuine_prob.setText('Genuine prob: -')
        self.lbl_infer_time.setText('推理耗时: 跳过')
        self.lbl_fps.setText(f"Display FPS: {self.display_fps:.1f} | Infer FPS: {self.infer_fps:.1f}")
        brightness_text = (
            f"Brightness: {result['brightness_mean']:.1f} | "
            f"Low-light: {'ON' if self.low_light_check.isChecked() else 'OFF'} | "
            f"Face crop: {'ON' if self.face_crop_check.isChecked() else 'OFF'} | 已阻止分类"
        )
        self.lbl_brightness.setStyleSheet('color: #c97a00;')
        self.lbl_brightness.setText(brightness_text)

        cv2.rectangle(display_frame, (12, 12), (display_frame.shape[1] - 12, display_frame.shape[0] - 12), color, 2)
        cv2.putText(
            display_frame,
            overlay,
            (24, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.76,
            color,
            2,
            cv2.LINE_AA,
        )
        self.current_depth_bgr = None
        self.depth_label.setText(f'伪深度图：{message}')
        self.depth_label.setPixmap(QPixmap())

    def show_frame_on_label(self, frame_bgr, label_widget):
        pixmap = cv2_bgr_to_qpixmap(frame_bgr, label_widget.size())
        label_widget.setPixmap(pixmap)

    def set_status(self, message):
        self.statusBar().showMessage(message)

    def closeEvent(self, event):
        self.close_camera()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = FASMainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
