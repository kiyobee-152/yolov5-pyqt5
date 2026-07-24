# YOLOv5 PyQt5 多路检测系统

这是一个基于 PyQt5 和 YOLOv5/ONNXRuntime 的多路目标检测应用，主要用于锚杆检测与异常目标识别。项目提供了图像、视频和摄像头输入的检测界面，支持同时管理 8 个检测画面，并可导出统计结果与报告。

## 功能特点

- 支持 8 路独立检测画面
- 支持图片、视频、摄像头输入
- 支持模型选择、置信度和 IoU 调整
- 支持检测结果实时显示与保存
- 支持检测记录统计、历史记录浏览
- 支持导出 CSV / JSON / HTML 报告
- 支持图像增强开关

## 项目结构

- main.py：主界面程序入口
- model_interface.py：模型接口与检测器封装
- video_processor.py：视频流处理与帧率控制
- post_processor.py：后处理逻辑
- statistics_panel.py：统计面板
- report_generator.py：HTML 报告生成
- detection_browser.py：历史记录浏览
- config_manager.py：配置管理
- logger.py：日志模块
- system_monitor.py：系统监控
- weights/：模型权重与类别文件
- results/：导出的结果与报告
- datasets/：数据集目录

## 环境要求

- Python 3.8+
- PyQt5
- OpenCV
- NumPy
- onnxruntime
- 可选：torch（如果使用 .pt / .pth 模型）

## 安装依赖

建议使用虚拟环境：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install opencv-python numpy PyQt5 onnxruntime
```

如果你要使用 PyTorch 权重文件，还需要安装：

```bash
pip install torch torchvision
```

## 运行方式

1. 确保模型文件已经放置在 weights 目录下，例如：
   - weights/best.onnx
   - weights/class_names.txt

2. 启动程序：

```bash
python main.py
```

## 使用说明

- 在主界面左侧选择模型、设备、置信度和 IoU
- 点击对应画面的“图片”“视频”“摄像头”按钮加载输入源
- 可通过“全部启动”同步开启所有通道
- 检测结果会在界面中实时显示，并可通过统计与报告功能查看汇总信息

## 配置文件

程序会自动生成并使用 config.json 保存用户配置，例如阈值、设备选择和增强设置。

## 注意事项

- 如果模型文件不存在，程序可能无法正常加载检测器
- 需要根据实际模型格式选择合适的权重文件
- 运行前建议确认摄像头或视频文件路径正确

## 许可证

本项目当前未声明许可证，请在使用前自行确认。
