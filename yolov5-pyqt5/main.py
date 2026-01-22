# -*- coding: utf-8 -*-
"""
基于深度学习的皮带传送带上锚杆检测系统
支持视频流处理、模型通用接口、交互界面和后处理功能
"""
import os
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog, QLabel, QApplication, QMessageBox
import image_rc
import threading
import cv2
import numpy as np
import time
from datetime import datetime
from model_interface import create_detector, BaseDetector
from video_processor import VideoProcessor, FrameRateController
from post_processor import PostProcessor


class Ui_MainWindow(QtWidgets.QMainWindow):
    signal = QtCore.pyqtSignal(str, str)

    def setupUi(self):
        self.setObjectName("MainWindow")
        self.resize(1400, 800)
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.detector = None
        self.weights_dir = './weights'
        
        # 初始化后处理器和视频处理器
        self.post_processor = PostProcessor()
        self.video_processor = VideoProcessor()
        self.frame_rate_controller = FrameRateController(target_fps=30)
        self.current_frame_id = 0
        self.save_detection_images = False

        self.picture = QtWidgets.QLabel(self.centralwidget)
        self.picture.setGeometry(QtCore.QRect(260, 10, 1010, 630))
        self.picture.setStyleSheet("background:black")
        self.picture.setObjectName("picture")
        self.picture.setScaledContents(True)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(10, 10, 81, 21))
        self.label_2.setObjectName("label_2")
        self.cb_weights = QtWidgets.QComboBox(self.centralwidget)
        self.cb_weights.setGeometry(QtCore.QRect(10, 40, 241, 21))
        self.cb_weights.setObjectName("cb_weights")
        self.cb_weights.currentIndexChanged.connect(self.cb_weights_changed)

        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(10, 70, 72, 21))
        self.label_3.setObjectName("label_3")
        self.hs_conf = QtWidgets.QSlider(self.centralwidget)
        self.hs_conf.setGeometry(QtCore.QRect(10, 100, 170, 18))
        self.hs_conf.setProperty("value", 25)
        self.hs_conf.setOrientation(QtCore.Qt.Horizontal)
        self.hs_conf.setObjectName("hs_conf")
        self.hs_conf.valueChanged.connect(self.conf_change)
        self.dsb_conf = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.dsb_conf.setGeometry(QtCore.QRect(190, 95, 61, 24))
        self.dsb_conf.setMaximum(1.0)
        self.dsb_conf.setSingleStep(0.01)
        self.dsb_conf.setProperty("value", 0.45)
        self.dsb_conf.setObjectName("dsb_conf")
        self.dsb_conf.valueChanged.connect(self.dsb_conf_change)
        self.dsb_iou = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.dsb_iou.setGeometry(QtCore.QRect(190, 155, 61, 24))
        self.dsb_iou.setMaximum(1.0)
        self.dsb_iou.setSingleStep(0.01)
        self.dsb_iou.setProperty("value", 0.45)
        self.dsb_iou.setObjectName("dsb_iou")
        self.dsb_iou.valueChanged.connect(self.dsb_iou_change)
        self.hs_iou = QtWidgets.QSlider(self.centralwidget)
        self.hs_iou.setGeometry(QtCore.QRect(10, 160, 170, 18))
        self.hs_iou.setProperty("value", 45)
        self.hs_iou.setOrientation(QtCore.Qt.Horizontal)
        self.hs_iou.setObjectName("hs_iou")
        self.hs_iou.valueChanged.connect(self.iou_change)
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(10, 130, 72, 21))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(10, 210, 90, 21))
        self.label_5.setObjectName("label_5")
        self.le_res = QtWidgets.QTextEdit(self.centralwidget)
        self.le_res.setGeometry(QtCore.QRect(10, 240, 241, 300))
        self.le_res.setObjectName("le_res")
        
        # 显示检测记录数量
        self.label_record_count = QtWidgets.QLabel(self.centralwidget)
        self.label_record_count.setGeometry(QtCore.QRect(10, 545, 241, 20))
        self.label_record_count.setObjectName("label_record_count")
        self.label_record_count.setText("检测记录: 0 条")
        # 使用简单的样式表
        try:
            self.label_record_count.setStyleSheet("color: gray;")
        except:
            pass
        
        # 添加后处理按钮组
        button_start_y = 570
        button_width = 115
        button_height = 28
        button_spacing = 35
        
        self.btn_save_image = QtWidgets.QPushButton(self.centralwidget)
        self.btn_save_image.setGeometry(QtCore.QRect(10, button_start_y, button_width, button_height))
        self.btn_save_image.setObjectName("btn_save_image")
        self.btn_save_image.setText("保存图像")
        self.btn_save_image.clicked.connect(self.save_current_image)
        self.btn_save_image.setStyleSheet("font-size:9pt;")
        
        self.btn_export_csv = QtWidgets.QPushButton(self.centralwidget)
        self.btn_export_csv.setGeometry(QtCore.QRect(130, button_start_y, button_width, button_height))
        self.btn_export_csv.setObjectName("btn_export_csv")
        self.btn_export_csv.setText("导出CSV")
        self.btn_export_csv.clicked.connect(self.export_csv)
        self.btn_export_csv.setStyleSheet("font-size:9pt;")
        
        self.btn_export_report = QtWidgets.QPushButton(self.centralwidget)
        self.btn_export_report.setGeometry(QtCore.QRect(10, button_start_y + button_spacing, button_width, button_height))
        self.btn_export_report.setObjectName("btn_export_report")
        self.btn_export_report.setText("导出报告")
        self.btn_export_report.clicked.connect(self.export_report)
        self.btn_export_report.setStyleSheet("font-size:9pt;")
        
        self.btn_export_json = QtWidgets.QPushButton(self.centralwidget)
        self.btn_export_json.setGeometry(QtCore.QRect(130, button_start_y + button_spacing, button_width, button_height))
        self.btn_export_json.setObjectName("btn_export_json")
        self.btn_export_json.setText("导出JSON")
        self.btn_export_json.clicked.connect(self.export_json)
        self.btn_export_json.setStyleSheet("font-size:9pt;")
        
        self.btn_clear_history = QtWidgets.QPushButton(self.centralwidget)
        self.btn_clear_history.setGeometry(QtCore.QRect(10, button_start_y + button_spacing * 2, 241, button_height))
        self.btn_clear_history.setObjectName("btn_clear_history")
        self.btn_clear_history.setText("清空检测记录")
        self.btn_clear_history.clicked.connect(self.clear_history)
        self.btn_clear_history.setStyleSheet("font-size:9pt;")
        
        # 视频预处理选项
        self.cb_enable_enhancement = QtWidgets.QCheckBox(self.centralwidget)
        self.cb_enable_enhancement.setGeometry(QtCore.QRect(10, button_start_y + button_spacing * 3 + 5, 241, 20))
        self.cb_enable_enhancement.setObjectName("cb_enable_enhancement")
        self.cb_enable_enhancement.setText("启用图像增强")
        self.cb_enable_enhancement.stateChanged.connect(self.toggle_enhancement)
        self.cb_enable_enhancement.setStyleSheet("font-size:9pt;")
        
        self.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1110, 30))
        self.menubar.setObjectName("menubar")
        self.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)
        self.toolBar = QtWidgets.QToolBar(self)
        self.toolBar.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.toolBar.setObjectName("toolBar")
        self.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.actionopenpic = QtWidgets.QAction(self)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/images/1.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionopenpic.setIcon(icon)
        self.actionopenpic.setObjectName("actionopenpic")
        self.actionopenpic.triggered.connect(self.open_image)
        self.action = QtWidgets.QAction(self)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/images/2.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action.setIcon(icon1)
        self.action.setObjectName("action")
        self.action.triggered.connect(self.open_video)
        self.action_2 = QtWidgets.QAction(self)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/images/3.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_2.setIcon(icon2)
        self.action_2.setObjectName("action_2")
        self.action_2.triggered.connect(self.open_camera)

        self.actionexit = QtWidgets.QAction(self)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/images/4.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionexit.setIcon(icon3)
        self.actionexit.setObjectName("actionexit")
        self.actionexit.triggered.connect(self.exit)
        
        # 添加菜单栏功能
        self.menu_file = QtWidgets.QMenu(self.menubar)
        self.menu_file.setTitle("文件")
        self.menubar.addAction(self.menu_file.menuAction())
        
        self.action_save_result = QtWidgets.QAction(self)
        self.action_save_result.setText("保存检测结果")
        self.action_save_result.triggered.connect(self.save_current_image)
        self.menu_file.addAction(self.action_save_result)
        
        self.action_export_json = QtWidgets.QAction(self)
        self.action_export_json.setText("导出JSON")
        self.action_export_json.triggered.connect(self.export_json)
        self.menu_file.addAction(self.action_export_json)
        
        self.menu_file.addSeparator()
        
        self.action_settings = QtWidgets.QAction(self)
        self.action_settings.setText("设置")
        self.menu_file.addAction(self.action_settings)

        self.toolBar.addAction(self.actionopenpic)
        self.toolBar.addAction(self.action)
        self.toolBar.addAction(self.action_2)
        self.toolBar.addAction(self.actionexit)

        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self)
        self.init_all()

    def exit(self):
        if self.camera_open:
            self.camera_open = False
            time.sleep(0.2)
        app = QApplication.instance()
        app.quit()

    def cb_weights_changed(self):
        if self.cb_weights.currentText() == "":
            return
        title = self.windowTitle()
        self.setWindowTitle('正在加载模型中..')
        try:
            weights_path = os.path.join(self.weights_dir, self.cb_weights.currentText())
            self.detector = create_detector(weights_path, model_type='auto',
                                           conf_thres=self.dsb_conf.value(),
                                           iou_thres=self.dsb_iou.value())
            self.setWindowTitle(title)
            if hasattr(self, 'ssl_show'):
                self.ssl_show.setText(f"模型加载成功: {self.cb_weights.currentText()}")
        except Exception as e:
            self.setWindowTitle(title)
            QMessageBox.warning(self, "错误", f"模型加载失败: {str(e)}")
            if hasattr(self, 'ssl_show'):
                self.ssl_show.setText(f"模型加载失败: {str(e)}")

    def open_camera(self):
        if self.action_2.text() == '打开摄像头':
            self.action_2.setText('停止')
            th = threading.Thread(target=self.start_camera)
            th.start()
        else:
            self.action_2.setText('打开摄像头')
            self.camera_open = False

    def set_res(self, res, flag):
        if flag == 'res':
            self.le_res.setPlainText(res)
            # 在主线程中更新记录数量显示（避免频繁调用）
            self.update_record_count()
        else:
            if hasattr(self, 'ssl_show'):
                self.ssl_show.setText(res)

    def init_all(self):
        self.signal.connect(self.set_res)
        self.image_path = None
        self.camera_open = False
        self.current_frame = None
        # 先初始化状态栏，确保 ssl_show 存在
        self.initStatusBar()
        # 然后加载权重列表（可能会触发 cb_weights_changed）
        self.load_weights_to_list()
        # 加载第一个模型
        if self.cb_weights.count() > 0:
            self.cb_weights_changed()
        self.beautify_left_panel()

    def resizeEvent(self, event):
        # 调整结果文本框高度
        self.le_res.setFixedHeight(self.height() - self.le_res.y() - self.statusbar.height() - 320)
        # 调整图像显示区域
        self.picture.setFixedSize(self.width() - self.le_res.x() - self.le_res.width() - 20,
                                  self.height() - self.statusbar.height())
        
        # 调整按钮位置（相对于结果文本框底部）
        button_start_y = self.le_res.y() + self.le_res.height() + 10
        
        if hasattr(self, 'label_record_count'):
            self.label_record_count.move(10, button_start_y)
            button_start_y += 25
        
        button_spacing = 35
        if hasattr(self, 'btn_save_image'):
            self.btn_save_image.move(10, button_start_y)
            if hasattr(self, 'btn_export_csv'):
                self.btn_export_csv.move(130, button_start_y)
            if hasattr(self, 'btn_export_report'):
                self.btn_export_report.move(10, button_start_y + button_spacing)
            if hasattr(self, 'btn_export_json'):
                self.btn_export_json.move(130, button_start_y + button_spacing)
            if hasattr(self, 'btn_clear_history'):
                self.btn_clear_history.move(10, button_start_y + button_spacing * 2)
            if hasattr(self, 'cb_enable_enhancement'):
                self.cb_enable_enhancement.move(10, button_start_y + button_spacing * 3 + 5)

    def initStatusBar(self):
        # 定义文本标签
        self.statusLabel = QLabel()
        self.statusLabel.setText("状态说明   ")
        self.statusLabel.setObjectName('statusLabel')

        self.ssl_show = QLabel()
        self.ssl_show.setText("等待操作中...")
        self.ssl_show.setFixedSize(600, 32)
        self.ssl_show.setObjectName('statusLabel_show')

        # 往状态栏中添加组件（stretch应该是拉伸组件宽度）-->会添加到状态栏左侧
        self.statusbar.addWidget(self.statusLabel)
        self.statusbar.addWidget(self.ssl_show)

    def open_video(self):
        if self.action.text() == '选择视频':
            fileName, fileType = QtWidgets.QFileDialog.getOpenFileName(self, "选取文件", os.getcwd(),
                                                                       "Video File(*.mp4 *.avi *.flv)")
            if fileName == '':
                return
            self.action.setText('停止')
            th = threading.Thread(target=self.start_video, args=(fileName,))
            th.start()
        else:
            self.action.setText('选择视频')
            self.camera_open = False

    def start_camera(self, camera_index=0):
        if self.detector is None:
            QMessageBox.warning(self, "警告", "请先选择模型!")
            self.action_2.setText('打开摄像头')
            return
        
        self.signal.emit('正在检测摄像头中...','camera')
        cap = cv2.VideoCapture(camera_index)
        self.camera_open = True
        self.current_frame_id = 0
        
        while self.camera_open:
            ret, frame = cap.read()
            if not ret:
                self.action_2.setText('打开摄像头')
                self.camera_open = False
                self.signal.emit('摄像头检测已停止!', 'camera')
                break
            
            # 视频预处理
            frame = self.video_processor.process_frame(frame)
            
            # 检测
            result_lists = self.detector.inference_image(frame)
            
            # 记录检测结果
            self.post_processor.add_detection(result_lists, frame_id=self.current_frame_id)
            
            # 绘制结果
            frame = self.detector.draw_image(result_lists, frame)
            frame = self.add_alarm_overlay(frame, result_lists)
            
            # 更新显示
            res = self.get_result_str(result_lists)
            self.signal.emit(res, 'res')
            
            # 保存当前帧（用于保存功能）
            self.current_frame = frame.copy()
            
            # 显示
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], QImage.Format_RGB888)
            self.picture.setPixmap(QPixmap.fromImage(img))
            
            self.current_frame_id += 1
            self.frame_rate_controller.wait_if_needed()
        
        cap.release()
        self.action_2.setText('打开摄像头')
        self.camera_open = False
        self.signal.emit('摄像头检测已停止!', 'camera')
        self.picture.setPixmap(QPixmap(""))

    def start_video(self, video_file):
        if self.detector is None:
            QMessageBox.warning(self, "警告", "请先选择模型!")
            self.action.setText('选择视频')
            return
        
        self.signal.emit('正在检测视频中...', 'video')
        
        # 获取视频信息
        video_info = self.video_processor.get_video_info(video_file)
        if video_info:
            fps = video_info.get('fps', 30)
            self.frame_rate_controller.set_fps(fps)
            info_text = f"视频信息: {video_info['width']}x{video_info['height']}, {fps}fps, {video_info.get('duration', 0):.1f}秒"
            if hasattr(self, 'ssl_show'):
                self.ssl_show.setText(info_text)
        
        cap = cv2.VideoCapture(video_file)
        self.camera_open = True
        self.current_frame_id = 0
        
        while self.camera_open:
            ret, frame = cap.read()
            if not ret:
                self.action.setText('选择视频')
                self.camera_open = False
                self.signal.emit('视频检测已停止!', 'video')
                break

            # 视频预处理
            frame = self.video_processor.process_frame(frame)
            
            # 检测
            result_lists = self.detector.inference_image(frame)
            
            # 记录检测结果
            self.post_processor.add_detection(result_lists, frame_id=self.current_frame_id)
            
            # 绘制结果
            frame = self.detector.draw_image(result_lists, frame)
            frame = self.add_alarm_overlay(frame, result_lists)
            
            # 更新显示
            res = self.get_result_str(result_lists)
            self.signal.emit(res, 'res')
            
            # 保存当前帧
            self.current_frame = frame.copy()
            
            # 显示
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], QImage.Format_RGB888)
            self.picture.setPixmap(QPixmap.fromImage(img))
            
            self.current_frame_id += 1
            self.frame_rate_controller.wait_if_needed()
        
        cap.release()
        self.action.setText('选择视频')
        self.camera_open = False
        self.signal.emit('视频检测已停止!', 'video')
        self.picture.setPixmap(QPixmap(""))

    def open_image(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "",
                                                       "图片文件 (*.jpg *.jpeg *.bmp *.png);;All Files(*)")
        if imgName == "":
            return
        self.image_path = imgName
        if hasattr(self, 'ssl_show'):
            self.ssl_show.setText("正在检测{}".format(os.path.basename(imgName)))
        threading.Thread(target=self.start_image).start()

    def get_result_str(self, result_list):
        result_dict = {}
        for result in result_list:
            name = result[0]
            if name in result_dict.keys():
                result_dict[name] += 1
            else:
                result_dict[name] = 1
        
        # 添加总体统计信息
        stats = self.post_processor.get_statistics()
        res = '当前帧检测结果:\n'
        res += '-' * 20 + '\n'
        for k, v in result_dict.items():
            res += f"{k}: {v}\n"
        
        # 添加累计统计
        if stats:
            res += '\n累计统计:\n'
            res += '-' * 20 + '\n'
            total = sum(stats.values())
            for k, v in stats.items():
                percentage = (v / total * 100) if total > 0 else 0
                res += f"{k}: {v} ({percentage:.1f}%)\n"
        
        # 不在get_result_str中更新UI，避免频繁调用导致卡顿
        # UI更新应该在主线程中通过信号机制进行
        
        return res
    
    def update_record_count(self):
        """更新检测记录数量显示"""
        if not hasattr(self, 'label_record_count'):
            return
        
        count = len(self.post_processor.detection_history)
        self.label_record_count.setText(f"检测记录: {count} 条")
        
        # 使用更简单的样式表，避免解析错误
        try:
            if count > 0:
                self.label_record_count.setStyleSheet("color: green; font-weight: bold;")
            else:
                self.label_record_count.setStyleSheet("color: gray;")
        except:
            # 如果样式表设置失败，至少更新文本
            pass

    def start_image(self):
        if self.detector is None:
            QMessageBox.warning(self, "警告", "请先选择模型!")
            return
        
        frame = cv2.imread(self.image_path)
        if frame is None:
            QMessageBox.warning(self, "错误", "无法读取图像文件!")
            return

        # 视频预处理（可选）
        frame = self.video_processor.process_frame(frame)
        
        # 检测
        result_lists = self.detector.inference_image(frame)
        
        # 记录检测结果
        self.post_processor.add_detection(result_lists, frame_id=0)
        
        # 绘制结果
        frame = self.detector.draw_image(result_lists, frame)
        frame = self.add_alarm_overlay(frame, result_lists)
        
        # 保存当前帧
        self.current_frame = frame.copy()
        
        # 显示（保持宽高比）
        height, width = frame.shape[:2]
        max_width = self.picture.width()
        max_height = self.picture.height()
        scale = min(max_width / width, max_height / height, 1.0)
        new_width = int(width * scale)
        new_height = int(height * scale)
        frame_resized = cv2.resize(frame, (new_width, new_height))
        
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        img = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], QImage.Format_RGB888)
        self.picture.setPixmap(QPixmap.fromImage(img))
        
        res = self.get_result_str(result_lists)
        self.signal.emit(res, 'res')
        if hasattr(self, 'ssl_show'):
            self.ssl_show.setText("检测已完成!")

    def iou_change(self):
        value = self.hs_iou.value() / 100
        self.dsb_iou.setValue(value)

    def conf_change(self):
        value = self.hs_conf.value() / 100
        self.dsb_conf.setValue(value)

    def dsb_iou_change(self):
        value = int(self.dsb_iou.value() * 100)
        self.hs_iou.setValue(value)
        if self.detector is not None:
            self.detector.set_iou(self.dsb_iou.value())

    def dsb_conf_change(self):
        value = int(self.dsb_conf.value() * 100)
        self.hs_conf.setValue(value)
        if self.detector is not None:
            self.detector.set_confidence(self.dsb_conf.value())

    def load_weights_to_list(self):
        for file in os.listdir(self.weights_dir):
            if file.endswith(".onnx") or file.endswith(".pt"):
                self.cb_weights.addItem(file)

    def add_alarm_overlay(self, frame, result_lists):
        if not result_lists:
            return frame
        
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # 统计当前帧的检测结果（只统计类别和数量）
        result_dict = {}
        for result in result_lists:
            name = result[0]
            if name in result_dict.keys():
                result_dict[name] += 1
            else:
                result_dict[name] = 1
        
        # 构建简洁的检测摘要（只显示类别和数量）
        detection_items = []
        for k, v in result_dict.items():
            detection_items.append(f"{k}:{v}")
        
        if detection_items:
            result_summary = ", ".join(detection_items)
        else:
            result_summary = "No detection"
        
        # 构建告警文本
        alarm_text = f"{timestamp} ALARM: {result_summary}"
        
        # 绘制半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 50), (0, 0, 0), thickness=-1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # 绘制告警文本（如果文本太长，可能需要分行显示）
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        color = (0, 0, 255)  # 红色
        
        # 计算文本大小，如果太长则分行
        text_size = cv2.getTextSize(alarm_text, font, font_scale, thickness)[0]
        max_width = frame.shape[1] - 20
        
        if text_size[0] > max_width:
            # 分行显示
            lines = []
            current_line = ""
            words = alarm_text.split()
            
            for word in words:
                test_line = current_line + (" " if current_line else "") + word
                test_size = cv2.getTextSize(test_line, font, font_scale, thickness)[0]
                if test_size[0] <= max_width:
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
            if current_line:
                lines.append(current_line)
            
            # 绘制多行文本
            y_offset = 25
            for i, line in enumerate(lines[:2]):  # 最多显示2行
                cv2.putText(frame, line, (10, y_offset + i * 25), font, font_scale, color, thickness, cv2.LINE_AA)
        else:
            # 单行显示
            cv2.putText(frame, alarm_text, (10, 35), font, font_scale, color, thickness, cv2.LINE_AA)
        
        return frame

    def beautify_left_panel(self):
        small_font = "font-size:10pt;"
        label_font = "font-size:10pt; font-weight:500;"
        self.label_2.setStyleSheet(label_font)
        self.label_3.setStyleSheet(label_font)
        self.label_4.setStyleSheet(label_font)
        self.label_5.setStyleSheet(label_font)
        self.cb_weights.setStyleSheet(small_font)
        self.dsb_conf.setStyleSheet(small_font)
        self.dsb_iou.setStyleSheet(small_font)
        self.le_res.setStyleSheet("font-size:10pt; padding:4px;")
        self.hs_conf.setStyleSheet("margin-left:2px; margin-right:2px;")
        self.hs_iou.setStyleSheet("margin-left:2px; margin-right:2px;")

    def save_current_image(self):
        """保存当前检测图像"""
        if self.current_frame is None:
            QMessageBox.warning(self, "警告", "没有可保存的图像!")
            return
        
        # 默认保存到results/images文件夹
        default_dir = os.path.join(self.post_processor.output_dir, 'images')
        os.makedirs(default_dir, exist_ok=True)
        default_filename = os.path.join(default_dir, 
                                       f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        
        filename, selected_filter = QFileDialog.getSaveFileName(
            self, 
            "保存图像", 
            default_filename,
            "JPEG图像 (*.jpg *.jpeg);;PNG图像 (*.png);;BMP图像 (*.bmp);;所有文件 (*.*)"
        )
        
        if filename:
            try:
                # 确保文件扩展名正确
                if not any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp']):
                    # 根据选择的过滤器添加扩展名
                    if 'JPEG' in selected_filter or 'jpg' in selected_filter.lower():
                        if not filename.lower().endswith(('.jpg', '.jpeg')):
                            filename += '.jpg'
                    elif 'PNG' in selected_filter:
                        if not filename.lower().endswith('.png'):
                            filename += '.png'
                    elif 'BMP' in selected_filter:
                        if not filename.lower().endswith('.bmp'):
                            filename += '.bmp'
                    else:
                        # 默认使用jpg
                        filename += '.jpg'
                
                # 确保目录存在
                save_dir = os.path.dirname(filename)
                if save_dir and not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                
                # 保存图像（BGR格式）
                success = cv2.imwrite(filename, self.current_frame)
                
                if success:
                    QMessageBox.information(self, "成功", f"图像已保存到:\n{filename}")
                else:
                    QMessageBox.warning(self, "错误", f"保存图像失败!\n请检查文件路径和权限。")
                    
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存图像时发生错误:\n{str(e)}")
    
    def export_report(self):
        """导出检测报告"""
        if not self.post_processor.detection_history:
            QMessageBox.warning(self, "警告", "没有检测记录可导出!\n请先进行检测操作。")
            return
        
        # 默认保存到results文件夹
        default_filename = os.path.join(self.post_processor.output_dir, 
                                       f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        filename, _ = QFileDialog.getSaveFileName(self, "导出报告", default_filename, "文本文件 (*.txt)")
        if filename:
            filepath = self.post_processor.export_report(filename)
            QMessageBox.information(self, "成功", f"报告已导出到:\n{filepath}\n\n共导出 {len(self.post_processor.detection_history)} 条检测记录")
    
    def export_csv(self):
        """导出CSV文件"""
        if not self.post_processor.detection_history:
            QMessageBox.warning(self, "警告", "没有检测记录可导出!\n请先进行检测操作。")
            return
        
        # 默认保存到results文件夹
        default_filename = os.path.join(self.post_processor.output_dir, 
                                       f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        filename, _ = QFileDialog.getSaveFileName(self, "导出CSV", default_filename, "CSV文件 (*.csv)")
        if filename:
            filepath = self.post_processor.export_csv(filename)
            QMessageBox.information(self, "成功", f"CSV已导出到:\n{filepath}\n\n共导出 {len(self.post_processor.detection_history)} 条检测记录")
    
    def export_json(self):
        """导出JSON文件"""
        if not self.post_processor.detection_history:
            QMessageBox.warning(self, "警告", "没有检测记录可导出!\n请先进行检测操作。")
            return
        
        # 默认保存到results文件夹
        default_filename = os.path.join(self.post_processor.output_dir, 
                                       f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        filename, _ = QFileDialog.getSaveFileName(self, "导出JSON", default_filename, "JSON文件 (*.json)")
        if filename:
            filepath = self.post_processor.export_json(filename)
            QMessageBox.information(self, "成功", f"JSON已导出到:\n{filepath}\n\n共导出 {len(self.post_processor.detection_history)} 条检测记录")
    
    def clear_history(self):
        """清空检测历史"""
        reply = QMessageBox.question(self, "确认", "确定要清空所有检测记录吗?", 
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.post_processor.clear_history()
            self.le_res.clear()
            self.update_record_count()
            QMessageBox.information(self, "成功", "检测记录已清空!")
    
    def toggle_enhancement(self, state):
        """切换图像增强"""
        self.video_processor.enable_enhancement = (state == QtCore.Qt.Checked)
    
    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("MainWindow", "基于深度学习的皮带传送带锚杆检测系统"))
        self.picture.setText(_translate("MainWindow", "TextLabel"))
        self.label_2.setText(_translate("MainWindow",
                                        "<html><head/><body><p><span style=\" font-size:7pt;\">模型选择</span></p></body></html>"))
        self.label_3.setText(_translate("MainWindow",
                                        "<html><head/><body><p><span style=\" font-size:7pt;\">置信度</span></p></body></html>"))
        self.label_4.setText(_translate("MainWindow",
                                        "<html><head/><body><p><span style=\" font-size:7pt;\">IOU</span></p></body></html>"))
        self.label_5.setText(_translate("MainWindow",
                                        "<html><head/><body><p><span style=\" font-size:7pt;\">结果统计</span></p></body></html>"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))
        self.actionopenpic.setText(_translate("MainWindow", "选择图片"))
        self.action.setText(_translate("MainWindow", "选择视频"))
        self.action_2.setText(_translate("MainWindow", "打开摄像头"))
        self.actionexit.setText(_translate("MainWindow", "退出程序"))


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    ui.setupUi()
    ui.show()
    sys.exit(app.exec_())
