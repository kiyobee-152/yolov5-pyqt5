# YOLOv5 PyQt5 多路检测系统

这是一个基于 PyQt5 和 YOLOv5/ONNXRuntime 的多路目标检测应用，主要用于**皮带输送机锚杆检测系统**，能够识别传送带上的**锚杆 (bolt)**、**大块煤 (large_sized_coal)** 和 **其他杂物 (Other_garbage)**。项目提供了图像、视频和摄像头输入的检测界面，支持同时管理 8 个检测画面，并可导出统计结果与报告。

## 功能特点

- 支持 8 路独立检测画面
- 支持图片、视频、摄像头输入
- 支持模型选择、置信度和 IoU 调整
- 支持检测结果实时显示与保存
- 支持检测记录统计、历史记录浏览
- 支持导出 CSV / JSON / HTML 报告
- 支持图像增强开关 (如 CLAHE、亮度、对比度调整)

## 项目结构

- **main.py**: 应用程序的主入口，负责构建 PyQt5 用户界面，协调和管理多达8个视频处理线程，并连接各个模块（如模型接口、处理器等）。核心应用逻辑在此文件。
- **model_interface.py**: AI 逻辑的核心。它为检测模型定义了一个清晰的抽象层（使用工厂模式）。当前实现了基于 ONNX 推理的 `YOLOv5ONNXDetector`，封装了完整的检测流程（预处理、推理、后处理）。所有模型检测逻辑的修改都应在此处进行。
- **yolov5_utils.py**: 包含 YOLOv5 推理管线中关键的底层辅助函数。这些函数源自官方 YOLOv5 仓库，对于图像预处理 (`letterbox`) 和后处理 (`non_max_suppression`, `scale_coords`) 至关重要，并被 `model_interface.py` 中的检测器调用。
- **video_processor.py**: 负责每个视频帧的预处理（在推理之前），特别是图像增强功能（CLAHE、亮度、对比度）。它还包含 `FrameRateController` 工具，对于管理视频播放速度和资源使用至关重要。
- **post_processor.py**: 作为中央的、线程安全的数据聚合器，收集所有并发检测结果。它负责存储、计数，并将结果导出为多种格式（CSV、JSON、TXT）。
- **report_generator.py**: 负责创建丰富、交互式的 HTML 报告。它展示了如何消费 `PostProcessor` 中的数据，生成具有过滤和分页等高级功能的报告。
- **statistics_panel.py**: 统计面板的界面逻辑。
- **detection_browser.py**: 历史记录浏览的界面逻辑。
- **config_manager.py**: 负责程序的配置管理，例如加载和保存用户设置。
- **logger.py**: 程序的日志记录模块。
- **system_monitor.py**: 提供实时的系统性能监控功能，如 GPU 使用情况。
- **weights/**: 存放模型权重文件和类别名称文件。
- **results/**: 存放导出的检测结果和报告。
- **datasets/**: 数据集目录，通常包含训练、验证和测试数据。

## 环境要求

- Python 3.8+
- PyQt5
- OpenCV (opencv-python)
- NumPy
- ONNX Runtime (onnxruntime)
- `requirements.txt` 中列出的所有依赖项

## 安装依赖

**强烈建议使用虚拟环境**：

1.  创建并激活虚拟环境：
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
    ```
2.  安装核心依赖：
    ```bash
    pip install -U pip
    pip install opencv-python numpy PyQt5 onnxruntime
    ```
3.  生成并安装 `requirements.txt` (推荐):
    ```bash
    # 在项目根目录下运行，确保所有当前环境的依赖被记录
    pip freeze > requirements.txt
    pip install -r requirements.txt
    ```

**如果你打算使用 PyTorch 格式的权重文件 (.pt / .pth) 或者需要 PyTorch 相关功能，还需要安装：**

```bash
pip install torch torchvision
```

## 运行方式

1.  **准备模型文件**: 确保你已将模型文件（例如 `best.onnx` 或 `yolov5s.pt`）和对应的类别名称文件 (`class_names.txt`) 放置在 `weights/` 目录下。
2.  **启动程序**:
    ```bash
    python main.py
    ```

## 使用说明

- 在主界面左侧选择所需的模型、推理设备（CPU/GPU）、置信度阈值和 IoU 阈值。
- 通过点击对应画面的“图片”、“视频”或“摄像头”按钮，加载不同的输入源。
- 可以使用“全部启动”按钮同步开启所有检测通道。
- 检测结果将实时显示在界面上，并通过统计与报告功能查看汇总信息。

## 配置文件

程序会自动在项目根目录下生成并使用 `config.json` 文件来保存用户的各项配置，包括但不限于阈值设置、设备选择和图像增强选项。

## 注意事项

- 如果模型文件路径不正确或文件不存在，程序可能无法正常初始化检测器。
- 请根据你使用的模型格式（ONNX 或 PyTorch）选择并准备相应的权重文件。
- 运行前，请务必确认摄像头设备或视频文件的路径是正确的。

## 许可证

**重要提示**: 本项目目前未明确声明任何开源许可证。在分发、修改或商业使用本项目之前，请务必自行确认并添加适当的许可证信息。