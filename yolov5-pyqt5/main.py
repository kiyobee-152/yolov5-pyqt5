# -*- coding: utf-8 -*-
"""
基于深度学习的皮带传送带上锚杆检测系统
支持视频流处理、模型通用接口、交互界面和后处理功能

本文件是整个系统的主程序入口，实现了一个基于 PyQt5 的桌面 GUI 应用。
核心类 Ui_MainWindow 继承自 QMainWindow，负责：
  1. 界面搭建：手动布局（非 Qt Designer 生成），构建左侧控制面板 + 右侧图像显示区域
  2. 模型管理：扫描 ./weights 目录下的模型文件，通过下拉框切换模型
  3. 输入源管理：支持三种输入源——图片文件、视频文件、摄像头实时流
  4. 检测流水线：预处理 → YOLOv5 推理 → 后处理记录 → 结果绘制 → 告警叠加 → 界面显示
  5. 结果导出：支持保存图像、导出 CSV/JSON/TXT 报告
  6. 参数调节：置信度和 IOU 阈值的实时滑块调节

多线程架构说明：
  - 视频/摄像头的检测循环运行在 threading.Thread 子线程中，避免阻塞 GUI 主线程
  - 子线程通过 pyqtSignal 信号机制向主线程发送数据，由主线程安全更新 UI 控件
  - camera_open 标志位用于主线程和子线程之间的停止信号通信

界面布局概览：
  ┌──────────────────────────────────────────────────────────┐
  │ 菜单栏: [文件] → 保存检测结果 / 导出JSON / 设置             │
  │ 工具栏: [选择图片] [选择视频] [打开摄像头] [退出程序]        │
  ├────────────┬─────────────────────────────────────────────┤
  │ 左侧控制面板│  右侧图像/视频显示区域                        │
  │ (宽250px)  │  (QLabel, 黑色背景, 自适应缩放)              │
  │            │                                            │
  │ · 模型选择  │                                             │
  │ · 置信度    │                                             │
  │ · IOU      │                                             │
  │ · 结果统计  │                                             │
  │ · 检测记录数│                                             │
  │ ·[保存图像] │                                             │
  │ ·[导出CSV]  │                                             │
  │ ·[导出报告] │                                             │
  │ ·[导出JSON] │                                             │
  │ ·[清空记录] │                                             │
  │ ·☐图像增强  │                                             │
  ├────────────┴─────────────────────────────────────────────┤
  │ 状态栏: [状态说明] [模型加载成功: yolov5s.onnx]            │
  └──────────────────────────────────────────────────────────┘

依赖关系：
  - model_interface.py：提供 create_detector 工厂函数和 BaseDetector 基类
  - video_processor.py：提供 VideoProcessor（帧预处理）和 FrameRateController（帧率控制）
  - post_processor.py：提供 PostProcessor（检测结果存储、统计、导出）
  - image_rc.py：PyQt5 资源文件，包含工具栏图标（由 pyrcc5 编译 .qrc 文件生成）
"""

# ---- 标准库导入 ----
import os       # 文件路径操作、目录扫描
import sys      # 命令行参数、程序退出
# ---- PyQt5 GUI 框架 ----
from PyQt5 import QtCore, QtGui, QtWidgets
# QThread: Qt 线程类（本项目实际使用 threading.Thread，QThread 已导入但未使用）
# pyqtSignal: Qt 信号类型定义（用于类级别信号声明）
from PyQt5.QtCore import QThread, pyqtSignal
# QImage: 将 numpy 图像数组转为 Qt 可显示的图像对象
# QPixmap: 用于在 QLabel 上显示图像
from PyQt5.QtGui import QImage, QPixmap
# QFileDialog: 文件打开/保存对话框
# QLabel: 文本/图像标签控件
# QApplication: Qt 应用程序实例
# QMessageBox: 弹窗对话框（警告、信息、确认等）
from PyQt5.QtWidgets import QFileDialog, QLabel, QApplication, QMessageBox
# 资源文件模块，包含工具栏按钮的图标资源（1.png~4.png）
# 由 pyrcc5 工具将 image.qrc 编译为 Python 模块
import image_rc
# ---- 标准库/第三方库 ----
import threading      # 多线程支持，用于在子线程中运行检测循环
import cv2            # OpenCV，图像/视频读写、色彩转换、绘图
import numpy as np    # numpy 数组操作
import time           # 时间格式化（告警叠加层时间戳）、sleep（退出等待）
from datetime import datetime  # 日期时间格式化（文件名生成）
# ---- 项目内部模块 ----
# create_detector: 工厂函数，根据模型文件自动创建检测器实例
# BaseDetector: 检测器抽象基类（类型提示用）
from model_interface import create_detector, BaseDetector
# VideoProcessor: 视频帧预处理（分辨率缩放、图像增强）
# FrameRateController: 帧率控制器（限制检测循环速度）
from video_processor import VideoProcessor, FrameRateController
# PostProcessor: 后处理器（检测结果存储、统计、多格式导出）
from post_processor import PostProcessor


# =============================================================================
# Ui_MainWindow - 主窗口类
# =============================================================================
# 继承 QMainWindow，是整个应用程序的核心类。
# 集成了界面构建、模型管理、检测流水线、结果展示和导出等全部功能。
class Ui_MainWindow(QtWidgets.QMainWindow):
    # ---- 类级别信号定义 ----
    # signal 用于子线程（检测循环）向主线程（GUI）安全传递数据
    # 参数1 (str): 要传递的文本内容（检测结果字符串或状态提示文字）
    # 参数2 (str): 标志位，'res' 表示更新结果文本框，其他值表示更新状态栏
    # PyQt5 要求信号必须在类级别定义，不能在 __init__ 中定义
    signal = QtCore.pyqtSignal(str, str)

    # =========================================================================
    # setupUi - 界面初始化方法
    # =========================================================================
    # 手动构建整个 GUI 界面（未使用 Qt Designer），包括：
    # - 左侧控制面板的所有控件（标签、下拉框、滑块、按钮等）
    # - 右侧图像显示区域
    # - 顶部工具栏和菜单栏
    # - 底部状态栏
    # 所有控件使用 setGeometry 进行绝对定位布局
    def setupUi(self):
        # 设置窗口对象名（Qt 内部标识）和初始大小 1400×800 像素
        self.setObjectName("MainWindow")
        self.resize(1400, 800)
        # 创建中心部件，所有控件都放置在此容器中
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        # 当前加载的检测器实例，初始为 None，在 cb_weights_changed() 中赋值
        self.detector = None
        # 模型权重文件目录，程序启动时会扫描此目录下的 .onnx 和 .pt 文件
        self.weights_dir = './weights'
        
        # ---- 初始化后处理器和视频处理器 ----
        # PostProcessor: 负责检测结果的存储、统计和导出，输出目录默认为 ./results/
        self.post_processor = PostProcessor()
        # VideoProcessor: 负责帧级预处理（分辨率缩放、图像增强），默认不做任何处理
        self.video_processor = VideoProcessor()
        # FrameRateController: 帧率控制器，默认 30fps，视频检测时会根据视频源 fps 动态调整
        self.frame_rate_controller = FrameRateController(target_fps=30)
        # 当前帧编号计数器，每处理一帧递增 1，用于 PostProcessor 记录帧ID
        self.current_frame_id = 0
        # 是否自动保存每帧检测图像的标志（当前未在 GUI 中暴露，预留功能）
        self.save_detection_images = False

        # =====================================================================
        # 右侧图像/视频显示区域
        # =====================================================================
        # QLabel 用于显示检测后的图像帧，通过 setPixmap 更新内容
        # 位置：x=260 y=10，宽 1010 高 630（会在 resizeEvent 中动态调整）
        self.picture = QtWidgets.QLabel(self.centralwidget)
        self.picture.setGeometry(QtCore.QRect(260, 10, 1010, 630))
        # 黑色背景：未加载图像时显示黑色
        self.picture.setStyleSheet("background:black")
        self.picture.setObjectName("picture")
        # setScaledContents(True): 图像自动拉伸填满整个 QLabel 区域
        self.picture.setScaledContents(True)

        # =====================================================================
        # 左侧控制面板 - 模型选择区域
        # =====================================================================
        # label_2: "模型选择" 标签文字
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(10, 10, 81, 21))
        self.label_2.setObjectName("label_2")
        # cb_weights: 模型权重文件下拉选择框
        # 在 load_weights_to_list() 中扫描 weights_dir 填充选项
        # 切换选项时触发 cb_weights_changed() 加载新模型
        self.cb_weights = QtWidgets.QComboBox(self.centralwidget)
        self.cb_weights.setGeometry(QtCore.QRect(10, 40, 241, 21))
        self.cb_weights.setObjectName("cb_weights")
        self.cb_weights.currentIndexChanged.connect(self.cb_weights_changed)

        # =====================================================================
        # 左侧控制面板 - 置信度调节区域
        # =====================================================================
        # label_3: "置信度" 标签文字
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(10, 70, 72, 21))
        self.label_3.setObjectName("label_3")
        # hs_conf: 置信度水平滑块，范围 0-100（整数），代表 0.00-1.00
        # 初始值 25，即 0.25（注意与 dsb_conf 的初始值 0.45 不一致，
        # 实际生效的是 dsb_conf 的值，因为 init_all 中最终由 dsb_conf 同步到模型）
        self.hs_conf = QtWidgets.QSlider(self.centralwidget)
        self.hs_conf.setGeometry(QtCore.QRect(10, 100, 170, 18))
        self.hs_conf.setProperty("value", 25)
        self.hs_conf.setOrientation(QtCore.Qt.Horizontal)
        self.hs_conf.setObjectName("hs_conf")
        # 滑块值变化时 → conf_change() → 同步更新 dsb_conf 数值框
        self.hs_conf.valueChanged.connect(self.conf_change)
        # dsb_conf: 置信度数值输入框，范围 0.00-1.00，步长 0.01
        # 与滑块双向联动：滑块变 → 更新数值框；数值框变 → 更新滑块 + 更新模型阈值
        self.dsb_conf = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.dsb_conf.setGeometry(QtCore.QRect(190, 95, 61, 24))
        self.dsb_conf.setMaximum(1.0)
        self.dsb_conf.setSingleStep(0.01)
        self.dsb_conf.setProperty("value", 0.45)
        self.dsb_conf.setObjectName("dsb_conf")
        # 数值框变化时 → dsb_conf_change() → 同步更新滑块 + 调用 detector.set_confidence()
        self.dsb_conf.valueChanged.connect(self.dsb_conf_change)

        # =====================================================================
        # 左侧控制面板 - IOU 调节区域
        # =====================================================================
        # dsb_iou: IOU 数值输入框，范围 0.00-1.00，步长 0.01，初始值 0.45
        self.dsb_iou = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.dsb_iou.setGeometry(QtCore.QRect(190, 155, 61, 24))
        self.dsb_iou.setMaximum(1.0)
        self.dsb_iou.setSingleStep(0.01)
        self.dsb_iou.setProperty("value", 0.45)
        self.dsb_iou.setObjectName("dsb_iou")
        self.dsb_iou.valueChanged.connect(self.dsb_iou_change)
        # hs_iou: IOU 水平滑块，范围 0-100，初始值 45（即 0.45）
        self.hs_iou = QtWidgets.QSlider(self.centralwidget)
        self.hs_iou.setGeometry(QtCore.QRect(10, 160, 170, 18))
        self.hs_iou.setProperty("value", 45)
        self.hs_iou.setOrientation(QtCore.Qt.Horizontal)
        self.hs_iou.setObjectName("hs_iou")
        self.hs_iou.valueChanged.connect(self.iou_change)
        # label_4: "IOU" 标签文字
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(10, 130, 72, 21))
        self.label_4.setObjectName("label_4")

        # =====================================================================
        # 左侧控制面板 - 结果统计区域
        # =====================================================================
        # label_5: "结果统计" 标签文字
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(10, 210, 90, 21))
        self.label_5.setObjectName("label_5")
        # le_res: 检测结果文本显示区域（QTextEdit，可滚动查看）
        # 显示当前帧检测结果 + 累计统计信息，由 signal → set_res() 在主线程中更新
        self.le_res = QtWidgets.QTextEdit(self.centralwidget)
        self.le_res.setGeometry(QtCore.QRect(10, 240, 241, 300))
        self.le_res.setObjectName("le_res")
        
        # =====================================================================
        # 左侧控制面板 - 检测记录数量标签
        # =====================================================================
        # 显示 PostProcessor 中累计的检测记录条数
        # 通过 update_record_count() 方法更新，有记录时显示绿色，无记录时显示灰色
        self.label_record_count = QtWidgets.QLabel(self.centralwidget)
        self.label_record_count.setGeometry(QtCore.QRect(10, 545, 241, 20))
        self.label_record_count.setObjectName("label_record_count")
        self.label_record_count.setText("检测记录: 0 条")
        # 使用简单的样式表，try-except 防御样式表解析异常
        try:
            self.label_record_count.setStyleSheet("color: gray;")
        except:
            pass
        
        # =====================================================================
        # 左侧控制面板 - 后处理功能按钮组
        # =====================================================================
        # 按钮布局参数：统一管理所有按钮的起始 y 坐标、宽高和间距
        button_start_y = 570     # 按钮组起始 y 坐标
        button_width = 115       # 单个按钮宽度
        button_height = 28       # 单个按钮高度
        button_spacing = 35      # 按钮行间距
        
        # 第一行左：保存图像按钮 → save_current_image()
        self.btn_save_image = QtWidgets.QPushButton(self.centralwidget)
        self.btn_save_image.setGeometry(QtCore.QRect(10, button_start_y, button_width, button_height))
        self.btn_save_image.setObjectName("btn_save_image")
        self.btn_save_image.setText("保存图像")
        self.btn_save_image.clicked.connect(self.save_current_image)
        self.btn_save_image.setStyleSheet("font-size:9pt;")
        
        # 第一行右：导出CSV按钮 → export_csv()
        self.btn_export_csv = QtWidgets.QPushButton(self.centralwidget)
        self.btn_export_csv.setGeometry(QtCore.QRect(130, button_start_y, button_width, button_height))
        self.btn_export_csv.setObjectName("btn_export_csv")
        self.btn_export_csv.setText("导出CSV")
        self.btn_export_csv.clicked.connect(self.export_csv)
        self.btn_export_csv.setStyleSheet("font-size:9pt;")
        
        # 第二行左：导出报告按钮 → export_report()
        self.btn_export_report = QtWidgets.QPushButton(self.centralwidget)
        self.btn_export_report.setGeometry(QtCore.QRect(10, button_start_y + button_spacing, button_width, button_height))
        self.btn_export_report.setObjectName("btn_export_report")
        self.btn_export_report.setText("导出报告")
        self.btn_export_report.clicked.connect(self.export_report)
        self.btn_export_report.setStyleSheet("font-size:9pt;")
        
        # 第二行右：导出JSON按钮 → export_json()
        self.btn_export_json = QtWidgets.QPushButton(self.centralwidget)
        self.btn_export_json.setGeometry(QtCore.QRect(130, button_start_y + button_spacing, button_width, button_height))
        self.btn_export_json.setObjectName("btn_export_json")
        self.btn_export_json.setText("导出JSON")
        self.btn_export_json.clicked.connect(self.export_json)
        self.btn_export_json.setStyleSheet("font-size:9pt;")
        
        # 第三行：清空检测记录按钮（全宽），→ clear_history()
        self.btn_clear_history = QtWidgets.QPushButton(self.centralwidget)
        self.btn_clear_history.setGeometry(QtCore.QRect(10, button_start_y + button_spacing * 2, 241, button_height))
        self.btn_clear_history.setObjectName("btn_clear_history")
        self.btn_clear_history.setText("清空检测记录")
        self.btn_clear_history.clicked.connect(self.clear_history)
        self.btn_clear_history.setStyleSheet("font-size:9pt;")
        
        # =====================================================================
        # 左侧控制面板 - 图像增强复选框
        # =====================================================================
        # 勾选后启用 VideoProcessor 的图像增强功能（亮度/对比度/饱和度调节）
        # 状态变化时触发 toggle_enhancement() 更新 video_processor.enable_enhancement 标志
        self.cb_enable_enhancement = QtWidgets.QCheckBox(self.centralwidget)
        self.cb_enable_enhancement.setGeometry(QtCore.QRect(10, button_start_y + button_spacing * 3 + 5, 241, 20))
        self.cb_enable_enhancement.setObjectName("cb_enable_enhancement")
        self.cb_enable_enhancement.setText("启用图像增强")
        self.cb_enable_enhancement.stateChanged.connect(self.toggle_enhancement)
        self.cb_enable_enhancement.setStyleSheet("font-size:9pt;")
        
        # =====================================================================
        # 设置中心部件、菜单栏、状态栏、工具栏
        # =====================================================================
        # 将 centralwidget 设为窗口的中心部件
        self.setCentralWidget(self.centralwidget)
        # 菜单栏：位于窗口最顶部
        self.menubar = QtWidgets.QMenuBar(self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1110, 30))
        self.menubar.setObjectName("menubar")
        self.setMenuBar(self.menubar)
        # 状态栏：位于窗口最底部，用于显示操作状态和模型信息
        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)
        # 工具栏：位于菜单栏下方，包含 4 个带图标的操作按钮
        # ToolButtonTextBesideIcon: 图标和文字并排显示
        self.toolBar = QtWidgets.QToolBar(self)
        self.toolBar.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.toolBar.setObjectName("toolBar")
        self.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)

        # =====================================================================
        # 工具栏按钮定义
        # =====================================================================
        # 按钮1：选择图片（图标 1.png）→ open_image()
        self.actionopenpic = QtWidgets.QAction(self)
        icon = QtGui.QIcon()
        # ":/images/1.png" 是 Qt 资源路径，对应 image_rc 中编译的图标资源
        icon.addPixmap(QtGui.QPixmap(":/images/1.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionopenpic.setIcon(icon)
        self.actionopenpic.setObjectName("actionopenpic")
        self.actionopenpic.triggered.connect(self.open_image)

        # 按钮2：选择视频（图标 2.png）→ open_video()
        self.action = QtWidgets.QAction(self)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/images/2.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action.setIcon(icon1)
        self.action.setObjectName("action")
        self.action.triggered.connect(self.open_video)

        # 按钮3：打开摄像头（图标 3.png）→ open_camera()
        self.action_2 = QtWidgets.QAction(self)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/images/3.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_2.setIcon(icon2)
        self.action_2.setObjectName("action_2")
        self.action_2.triggered.connect(self.open_camera)

        # 按钮4：退出程序（图标 4.png）→ exit()
        self.actionexit = QtWidgets.QAction(self)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/images/4.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionexit.setIcon(icon3)
        self.actionexit.setObjectName("actionexit")
        self.actionexit.triggered.connect(self.exit)
        
        # =====================================================================
        # 菜单栏 - "文件" 菜单
        # =====================================================================
        # 提供菜单栏方式的快捷操作入口，与左侧面板按钮功能重复，方便用户使用习惯
        self.menu_file = QtWidgets.QMenu(self.menubar)
        self.menu_file.setTitle("文件")
        self.menubar.addAction(self.menu_file.menuAction())
        
        # 菜单项："保存检测结果" → save_current_image()
        self.action_save_result = QtWidgets.QAction(self)
        self.action_save_result.setText("保存检测结果")
        self.action_save_result.triggered.connect(self.save_current_image)
        self.menu_file.addAction(self.action_save_result)
        
        # 菜单项："导出JSON" → export_json()
        self.action_export_json = QtWidgets.QAction(self)
        self.action_export_json.setText("导出JSON")
        self.action_export_json.triggered.connect(self.export_json)
        self.menu_file.addAction(self.action_export_json)
        
        # 分隔线
        self.menu_file.addSeparator()
        
        # 菜单项："设置"（预留功能，当前未绑定具体操作）
        self.action_settings = QtWidgets.QAction(self)
        self.action_settings.setText("设置")
        self.menu_file.addAction(self.action_settings)

        # 将 4 个 Action 按顺序添加到工具栏
        self.toolBar.addAction(self.actionopenpic)
        self.toolBar.addAction(self.action)
        self.toolBar.addAction(self.action_2)
        self.toolBar.addAction(self.actionexit)

        # 设置所有控件的显示文字（国际化翻译入口）
        self.retranslateUi()
        # 自动根据对象名称连接信号和槽（Qt 的命名约定机制）
        QtCore.QMetaObject.connectSlotsByName(self)
        # 执行初始化流程：连接信号、加载模型、美化界面
        self.init_all()

    # =========================================================================
    # exit - 退出程序
    # =========================================================================
    # 安全退出逻辑：先停止可能正在运行的摄像头/视频检测线程，
    # 等待 0.2 秒让子线程有时间释放资源（如 VideoCapture），再退出应用
    def exit(self):
        if self.camera_open:
            # 设置标志位通知子线程停止循环
            self.camera_open = False
            # 等待子线程响应停止信号并释放 VideoCapture 资源
            time.sleep(0.2)
        # 获取当前 QApplication 实例并退出
        app = QApplication.instance()
        app.quit()

    # =========================================================================
    # cb_weights_changed - 模型下拉框切换回调
    # =========================================================================
    # 当用户在模型下拉框中选择不同的模型文件时触发。
    # 使用 create_detector 工厂函数加载新模型，同时传入当前 GUI 上的置信度和 IOU 值。
    # 加载过程中临时修改窗口标题为 "正在加载模型中.."，完成后恢复原标题。
    def cb_weights_changed(self):
        # 空文本时（如下拉框初始化）忽略
        if self.cb_weights.currentText() == "":
            return
        # 保存当前窗口标题，加载完成后恢复
        title = self.windowTitle()
        self.setWindowTitle('正在加载模型中..')
        try:
            # 拼接完整的模型文件路径
            weights_path = os.path.join(self.weights_dir, self.cb_weights.currentText())
            # 调用工厂函数创建检测器实例
            # model_type='auto' 会根据文件扩展名自动选择 ONNX 或 PyTorch 检测器
            # conf_thres 和 iou_thres 从 GUI 控件获取当前值
            self.detector = create_detector(weights_path, model_type='auto',
                                           conf_thres=self.dsb_conf.value(),
                                           iou_thres=self.dsb_iou.value())
            # 加载成功，恢复标题并更新状态栏
            self.setWindowTitle(title)
            if hasattr(self, 'ssl_show'):
                self.ssl_show.setText(f"模型加载成功: {self.cb_weights.currentText()}")
        except Exception as e:
            # 加载失败，恢复标题并弹出错误对话框
            self.setWindowTitle(title)
            QMessageBox.warning(self, "错误", f"模型加载失败: {str(e)}")
            if hasattr(self, 'ssl_show'):
                self.ssl_show.setText(f"模型加载失败: {str(e)}")

    # =========================================================================
    # open_camera - 摄像头按钮点击回调
    # =========================================================================
    # 按钮文字在 "打开摄像头" 和 "停止" 之间切换，实现开启/停止的状态切换逻辑。
    # 开启时在新线程中启动 start_camera()，停止时设置 camera_open=False 通知子线程退出。
    def open_camera(self):
        if self.action_2.text() == '打开摄像头':
            # 切换按钮文字为"停止"
            self.action_2.setText('停止')
            # 在新线程中启动摄像头检测循环，避免阻塞 GUI 主线程
            th = threading.Thread(target=self.start_camera)
            th.start()
        else:
            # 用户点击"停止"，恢复按钮文字并设置停止标志
            self.action_2.setText('打开摄像头')
            self.camera_open = False

    # =========================================================================
    # set_res - 信号槽：接收子线程数据并更新 GUI
    # =========================================================================
    # 这是 signal 信号的槽函数，在 GUI 主线程中执行。
    # 子线程通过 self.signal.emit(text, flag) 发送数据，
    # 主线程自动调用此函数进行 UI 更新（线程安全）。
    #
    # 参数 flag 的含义：
    #   'res': 更新左侧面板的结果统计文本框 + 更新检测记录数量标签
    #   其他值（如 'camera', 'video'）: 更新底部状态栏的提示文字
    def set_res(self, res, flag):
        if flag == 'res':
            # 更新结果文本框内容
            self.le_res.setPlainText(res)
            # 在主线程中更新记录数量显示（避免频繁调用）
            self.update_record_count()
        else:
            # 更新状态栏提示文字
            if hasattr(self, 'ssl_show'):
                self.ssl_show.setText(res)

    # =========================================================================
    # init_all - 全局初始化方法
    # =========================================================================
    # 在 setupUi() 末尾调用，完成以下初始化工作：
    # 1. 连接信号到槽函数
    # 2. 初始化状态变量
    # 3. 初始化状态栏（必须先于模型加载，因为加载过程中需要更新状态栏）
    # 4. 扫描并加载模型文件列表
    # 5. 自动加载第一个模型
    # 6. 美化左侧面板样式
    def init_all(self):
        # 将 signal 信号连接到 set_res 槽函数
        self.signal.connect(self.set_res)
        # 初始化状态变量
        self.image_path = None       # 当前打开的图片路径
        self.camera_open = False     # 摄像头/视频是否正在运行的标志位
        self.current_frame = None    # 当前显示的帧（用于"保存图像"功能）
        # 先初始化状态栏，确保 ssl_show 存在
        # 重要：必须在 load_weights_to_list 之前调用，因为加载模型会尝试访问 ssl_show
        self.initStatusBar()
        # 然后加载权重列表（可能会触发 cb_weights_changed）
        self.load_weights_to_list()
        # 加载第一个模型（如果有的话）
        if self.cb_weights.count() > 0:
            self.cb_weights_changed()
        # 美化左侧面板的字体和样式
        self.beautify_left_panel()

    # =========================================================================
    # resizeEvent - 窗口尺寸变化事件处理
    # =========================================================================
    # 当用户拖拽调整窗口大小时，Qt 自动调用此方法。
    # 需要重新计算左侧面板各控件的位置和右侧图像区域的尺寸，实现简易的响应式布局。
    # 注意：由于使用绝对定位布局，窗口缩放时需要手动调整每个控件的位置。
    def resizeEvent(self, event):
        # 调整结果文本框高度：窗口高度 - 文本框顶部 y - 状态栏高度 - 预留给按钮的空间(320px)
        self.le_res.setFixedHeight(self.height() - self.le_res.y() - self.statusbar.height() - 320)
        # 调整图像显示区域：占满左侧面板右边的全部剩余空间
        self.picture.setFixedSize(self.width() - self.le_res.x() - self.le_res.width() - 20,
                                  self.height() - self.statusbar.height())
        
        # 调整按钮位置（相对于结果文本框底部动态定位）
        button_start_y = self.le_res.y() + self.le_res.height() + 10
        
        # 检测记录数量标签
        if hasattr(self, 'label_record_count'):
            self.label_record_count.move(10, button_start_y)
            button_start_y += 25
        
        # 按钮组逐行排列
        # hasattr 检查确保在 setupUi 完成前调用 resizeEvent 不会报错
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

    # =========================================================================
    # initStatusBar - 初始化底部状态栏
    # =========================================================================
    # 在状态栏左侧添加两个 QLabel：
    # 1. statusLabel: 固定文字 "状态说明"（装饰性标签）
    # 2. ssl_show: 动态提示文字，显示当前操作状态（如"模型加载成功"、"正在检测中"等）
    def initStatusBar(self):
        # 定义文本标签
        self.statusLabel = QLabel()
        self.statusLabel.setText("状态说明   ")
        self.statusLabel.setObjectName('statusLabel')

        # 动态状态标签：宽 600px，用于显示各种操作提示信息
        self.ssl_show = QLabel()
        self.ssl_show.setText("等待操作中...")
        self.ssl_show.setFixedSize(600, 32)
        self.ssl_show.setObjectName('statusLabel_show')

        # 往状态栏中添加组件（stretch应该是拉伸组件宽度）-->会添加到状态栏左侧
        self.statusbar.addWidget(self.statusLabel)
        self.statusbar.addWidget(self.ssl_show)

    # =========================================================================
    # open_video - 视频按钮点击回调
    # =========================================================================
    # 与 open_camera 类似的状态切换逻辑：
    # - "选择视频" 状态：弹出文件对话框选择视频文件，然后在新线程中启动 start_video()
    # - "停止" 状态：设置 camera_open=False 通知子线程退出
    def open_video(self):
        if self.action.text() == '选择视频':
            # 弹出文件选择对话框，支持 mp4/avi/flv 格式
            fileName, fileType = QtWidgets.QFileDialog.getOpenFileName(self, "选取文件", os.getcwd(),
                                                                       "Video File(*.mp4 *.avi *.flv)")
            if fileName == '':
                return
            self.action.setText('停止')
            # 在新线程中启动视频检测循环，传入视频文件路径
            th = threading.Thread(target=self.start_video, args=(fileName,))
            th.start()
        else:
            self.action.setText('选择视频')
            self.camera_open = False

    # =========================================================================
    # start_camera - 摄像头检测主循环（运行在子线程中）
    # =========================================================================
    # 完整的实时检测流水线：
    #   打开摄像头 → 循环{读帧 → 预处理 → 推理 → 记录 → 绘制 → 显示} → 释放摄像头
    #
    # 线程安全说明：
    # - camera_open 标志位由主线程写入（设为 False），子线程读取（检查是否继续循环）
    # - UI 更新通过 signal.emit() 发送到主线程处理
    # - QPixmap.fromImage() 虽然在子线程中调用，但 QLabel.setPixmap 是线程安全的
    def start_camera(self, camera_index=0):
        # 前置检查：是否已加载模型
        if self.detector is None:
            QMessageBox.warning(self, "警告", "请先选择模型!")
            self.action_2.setText('打开摄像头')
            return
        
        # 通过信号通知主线程更新状态栏
        self.signal.emit('正在检测摄像头中...','camera')
        # 打开摄像头（默认设备索引 0，即系统默认摄像头）
        cap = cv2.VideoCapture(camera_index)
        self.camera_open = True
        self.current_frame_id = 0
        
        # ---- 检测主循环 ----
        while self.camera_open:
            # 读取一帧图像
            ret, frame = cap.read()
            if not ret:
                # 读取失败（摄像头断开等），退出循环
                self.action_2.setText('打开摄像头')
                self.camera_open = False
                self.signal.emit('摄像头检测已停止!', 'camera')
                break
            
            # 步骤1：视频预处理（分辨率缩放、图像增强等，由 VideoProcessor 处理）
            frame = self.video_processor.process_frame(frame)
            
            # 步骤2：YOLOv5 模型推理，返回检测结果列表
            # 格式：[[class_name, confidence, x1, y1, x2, y2], ...]
            result_lists = self.detector.inference_image(frame)
            
            # 步骤3：将检测结果记录到后处理器（用于统计和导出）
            self.post_processor.add_detection(result_lists, frame_id=self.current_frame_id)
            
            # 步骤4：在帧上绘制检测框和标签
            frame = self.detector.draw_image(result_lists, frame)
            # 步骤5：在帧顶部叠加告警信息条（时间戳 + 检测摘要）
            frame = self.add_alarm_overlay(frame, result_lists)
            
            # 步骤6：生成结果文本并通过信号发送到主线程更新 GUI
            res = self.get_result_str(result_lists)
            self.signal.emit(res, 'res')
            
            # 保存当前帧的副本（用户点击"保存图像"按钮时保存此帧）
            # .copy() 确保不受后续帧覆盖的影响
            self.current_frame = frame.copy()
            
            # 步骤7：将 OpenCV BGR 图像转为 Qt 可显示的格式并更新到 QLabel
            # BGR → RGB 色彩转换（OpenCV 默认 BGR，Qt 显示需要 RGB）
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 创建 QImage 对象：传入像素数据、宽、高、格式
            img = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], QImage.Format_RGB888)
            # 转为 QPixmap 并设置到显示标签
            self.picture.setPixmap(QPixmap.fromImage(img))
            
            # 帧计数器递增
            self.current_frame_id += 1
            # 步骤8：帧率控制——如果处理速度过快则 sleep 等待，稳定帧率
            self.frame_rate_controller.wait_if_needed()
        
        # ---- 循环结束后的清理工作 ----
        # 释放摄像头资源
        cap.release()
        # 恢复按钮文字
        self.action_2.setText('打开摄像头')
        self.camera_open = False
        self.signal.emit('摄像头检测已停止!', 'camera')
        # 清空显示区域（显示黑色背景）
        self.picture.setPixmap(QPixmap(""))

    # =========================================================================
    # start_video - 视频文件检测主循环（运行在子线程中）
    # =========================================================================
    # 与 start_camera 的流程基本一致，区别在于：
    # 1. 输入源是视频文件而非摄像头
    # 2. 开始前会获取视频元信息（fps、分辨率、时长）并显示到状态栏
    # 3. 帧率控制器会根据视频源的 fps 动态调整，确保视频以原始速度播放
    def start_video(self, video_file):
        if self.detector is None:
            QMessageBox.warning(self, "警告", "请先选择模型!")
            self.action.setText('选择视频')
            return
        
        self.signal.emit('正在检测视频中...', 'video')
        
        # 获取视频元信息并显示
        video_info = self.video_processor.get_video_info(video_file)
        if video_info:
            fps = video_info.get('fps', 30)
            # 将帧率控制器的目标 fps 设为视频源的 fps，确保原速播放
            self.frame_rate_controller.set_fps(fps)
            # 在状态栏显示视频信息（分辨率、帧率、时长）
            info_text = f"视频信息: {video_info['width']}x{video_info['height']}, {fps}fps, {video_info.get('duration', 0):.1f}秒"
            if hasattr(self, 'ssl_show'):
                self.ssl_show.setText(info_text)
        
        # 打开视频文件
        cap = cv2.VideoCapture(video_file)
        self.camera_open = True
        self.current_frame_id = 0
        
        # ---- 检测主循环（与 start_camera 结构相同）----
        while self.camera_open:
            ret, frame = cap.read()
            if not ret:
                # 视频播放完毕或读取错误
                self.action.setText('选择视频')
                self.camera_open = False
                self.signal.emit('视频检测已停止!', 'video')
                break

            # 预处理 → 推理 → 记录 → 绘制 → 告警叠加 → GUI更新 → 显示
            frame = self.video_processor.process_frame(frame)
            result_lists = self.detector.inference_image(frame)
            self.post_processor.add_detection(result_lists, frame_id=self.current_frame_id)
            frame = self.detector.draw_image(result_lists, frame)
            frame = self.add_alarm_overlay(frame, result_lists)
            res = self.get_result_str(result_lists)
            self.signal.emit(res, 'res')
            self.current_frame = frame.copy()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], QImage.Format_RGB888)
            self.picture.setPixmap(QPixmap.fromImage(img))
            self.current_frame_id += 1
            self.frame_rate_controller.wait_if_needed()
        
        # 清理
        cap.release()
        self.action.setText('选择视频')
        self.camera_open = False
        self.signal.emit('视频检测已停止!', 'video')
        self.picture.setPixmap(QPixmap(""))

    # =========================================================================
    # open_image - 图片按钮点击回调
    # =========================================================================
    # 弹出文件对话框让用户选择一张图片，然后在子线程中执行 start_image() 进行检测。
    # 虽然图片检测是一次性操作（非循环），但仍放在子线程中以避免阻塞 GUI。
    def open_image(self):
        # 打开文件对话框，支持 jpg/jpeg/bmp/png 格式
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "",
                                                       "图片文件 (*.jpg *.jpeg *.bmp *.png);;All Files(*)")
        if imgName == "":
            return
        # 保存图片路径供 start_image() 使用
        self.image_path = imgName
        # 更新状态栏显示正在检测的文件名
        if hasattr(self, 'ssl_show'):
            self.ssl_show.setText("正在检测{}".format(os.path.basename(imgName)))
        # 在新线程中执行图片检测
        threading.Thread(target=self.start_image).start()

    # =========================================================================
    # get_result_str - 生成检测结果显示文本
    # =========================================================================
    # 将当前帧的检测结果和累计统计信息格式化为人类可读的多行文本字符串，
    # 用于显示在左侧面板的结果文本框中。
    def get_result_str(self, result_list):
        # ---- 统计当前帧各类别的检测数量 ----
        result_dict = {}
        for result in result_list:
            name = result[0]  # 类别名称
            if name in result_dict.keys():
                result_dict[name] += 1
            else:
                result_dict[name] = 1
        
        # ---- 从后处理器获取全局累计统计 ----
        stats = self.post_processor.get_statistics()
        
        # ---- 组合当前帧结果文本 ----
        res = '当前帧检测结果:\n'
        res += '-' * 20 + '\n'
        for k, v in result_dict.items():
            res += f"{k}: {v}\n"
        
        # ---- 追加累计统计文本（包含百分比） ----
        if stats:
            res += '\n累计统计:\n'
            res += '-' * 20 + '\n'
            total = sum(stats.values())
            for k, v in stats.items():
                percentage = (v / total * 100) if total > 0 else 0
                res += f"{k}: {v} ({percentage:.1f}%)\n"
        
        return res
    
    # =========================================================================
    # update_record_count - 更新检测记录数量标签
    # =========================================================================
    def update_record_count(self):
        """更新检测记录数量显示"""
        if not hasattr(self, 'label_record_count'):
            return
        
        count = len(self.post_processor.detection_history)
        self.label_record_count.setText(f"检测记录: {count} 条")
        
        try:
            if count > 0:
                self.label_record_count.setStyleSheet("color: green; font-weight: bold;")
            else:
                self.label_record_count.setStyleSheet("color: gray;")
        except:
            pass

    # =========================================================================
    # start_image - 图片检测处理（运行在子线程中）
    # =========================================================================
    def start_image(self):
        if self.detector is None:
            QMessageBox.warning(self, "警告", "请先选择模型!")
            return
        
        frame = cv2.imread(self.image_path)
        if frame is None:
            QMessageBox.warning(self, "错误", "无法读取图像文件!")
            return

        frame = self.video_processor.process_frame(frame)
        result_lists = self.detector.inference_image(frame)
        self.post_processor.add_detection(result_lists, frame_id=0)
        frame = self.detector.draw_image(result_lists, frame)
        frame = self.add_alarm_overlay(frame, result_lists)
        self.current_frame = frame.copy()
        
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
        result_dict = {}
        for result in result_lists:
            name = result[0]
            if name in result_dict.keys():
                result_dict[name] += 1
            else:
                result_dict[name] = 1
        
        detection_items = []
        for k, v in result_dict.items():
            detection_items.append(f"{k}:{v}")
        
        result_summary = ", ".join(detection_items) if detection_items else "No detection"
        alarm_text = f"{timestamp} ALARM: {result_summary}"
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 50), (0, 0, 0), thickness=-1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        color = (0, 0, 255)
        
        text_size = cv2.getTextSize(alarm_text, font, font_scale, thickness)[0]
        max_width = frame.shape[1] - 20
        
        if text_size[0] > max_width:
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
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f7fb;
            }
            QLabel {
                color: #2d3748;
                font-size: 10pt;
            }
            QComboBox, QDoubleSpinBox, QTextEdit, QPushButton {
                border: 1px solid #d8dee9;
                border-radius: 6px;
                background-color: #ffffff;
                font-size: 10pt;
            }
            QComboBox, QDoubleSpinBox {
                min-height: 26px;
                padding: 0 8px;
            }
            QTextEdit {
                padding: 8px;
                background: #ffffff;
                selection-background-color: #4f86f7;
            }
            QPushButton {
                min-height: 28px;
                padding: 0 8px;
                color: #1f2937;
            }
            QPushButton:hover {
                background-color: #eef4ff;
                border-color: #93b4ff;
            }
            QPushButton:pressed {
                background-color: #dfe9ff;
            }
            QSlider::groove:horizontal {
                height: 5px;
                border-radius: 2px;
                background: #dbe4f3;
            }
            QSlider::handle:horizontal {
                width: 14px;
                margin: -5px 0;
                border-radius: 7px;
                background: #4f86f7;
                border: 1px solid #3f74e6;
            }
            QCheckBox {
                spacing: 6px;
                color: #2d3748;
                font-size: 10pt;
            }
            QCheckBox::indicator {
                width: 15px;
                height: 15px;
            }
            QStatusBar {
                background-color: #eef2f9;
                color: #2d3748;
            }
            QToolBar {
                background-color: #eef2f9;
                border: none;
                spacing: 5px;
                padding: 4px;
            }
            QMenuBar {
                background-color: #ffffff;
            }
            QMenuBar::item:selected {
                background: #eef4ff;
            }
        """)

        label_font = "font-size:10pt; font-weight:600; color:#1f2937;"
        self.label_2.setStyleSheet(label_font)
        self.label_3.setStyleSheet(label_font)
        self.label_4.setStyleSheet(label_font)
        self.label_5.setStyleSheet(label_font)
        self.le_res.setStyleSheet("font-size:10pt; padding:8px; background:#ffffff;")
        self.picture.setStyleSheet("background:#0f172a; border:1px solid #d8dee9; border-radius:8px;")

        self.btn_save_image.setStyleSheet("font-size:10pt; font-weight:500;")
        self.btn_export_csv.setStyleSheet("font-size:10pt; font-weight:500;")
        self.btn_export_report.setStyleSheet("font-size:10pt; font-weight:500;")
        self.btn_export_json.setStyleSheet("font-size:10pt; font-weight:500;")
        self.btn_clear_history.setStyleSheet("font-size:10pt; font-weight:500;")

    def save_current_image(self):
        """保存当前检测图像"""
        if self.current_frame is None:
            QMessageBox.warning(self, "警告", "没有可保存的图像!")
            return
        
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
                if not any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp']):
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
                        filename += '.jpg'
                
                save_dir = os.path.dirname(filename)
                if save_dir and not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                
                success = cv2.imwrite(filename, self.current_frame)
                if success:
                    QMessageBox.information(self, "成功", f"图像已保存到:\n{filename}")
                else:
                    QMessageBox.warning(self, "错误", f"保存图像失败!\n请检查文件路径和权限。")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存图像时发生错误:\n{str(e)}")
    
    def export_report(self):
        if not self.post_processor.detection_history:
            QMessageBox.warning(self, "警告", "没有检测记录可导出!\n请先进行检测操作。")
            return
        default_filename = os.path.join(self.post_processor.output_dir, 
                                       f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        filename, _ = QFileDialog.getSaveFileName(self, "导出报告", default_filename, "文本文件 (*.txt)")
        if filename:
            filepath = self.post_processor.export_report(filename)
            QMessageBox.information(self, "成功", f"报告已导出到:\n{filepath}\n\n共导出 {len(self.post_processor.detection_history)} 条检测记录")
    
    def export_csv(self):
        if not self.post_processor.detection_history:
            QMessageBox.warning(self, "警告", "没有检测记录可导出!\n请先进行检测操作。")
            return
        default_filename = os.path.join(self.post_processor.output_dir, 
                                       f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        filename, _ = QFileDialog.getSaveFileName(self, "导出CSV", default_filename, "CSV文件 (*.csv)")
        if filename:
            filepath = self.post_processor.export_csv(filename)
            QMessageBox.information(self, "成功", f"CSV已导出到:\n{filepath}\n\n共导出 {len(self.post_processor.detection_history)} 条检测记录")
    
    def export_json(self):
        if not self.post_processor.detection_history:
            QMessageBox.warning(self, "警告", "没有检测记录可导出!\n请先进行检测操作。")
            return
        default_filename = os.path.join(self.post_processor.output_dir, 
                                       f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        filename, _ = QFileDialog.getSaveFileName(self, "导出JSON", default_filename, "JSON文件 (*.json)")
        if filename:
            filepath = self.post_processor.export_json(filename)
            QMessageBox.information(self, "成功", f"JSON已导出到:\n{filepath}\n\n共导出 {len(self.post_processor.detection_history)} 条检测记录")
    
    def clear_history(self):
        reply = QMessageBox.question(self, "确认", "确定要清空所有检测记录吗?", 
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.post_processor.clear_history()
            self.le_res.clear()
            self.update_record_count()
            QMessageBox.information(self, "成功", "检测记录已清空!")
    
    def toggle_enhancement(self, state):
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
