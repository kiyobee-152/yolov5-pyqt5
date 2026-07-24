# ConveyorBolt Detector — 基于 YOLOv5 的多路输送带锚杆检测平台

一个面向工业现场的、可视化且可部署的多路目标检测桌面应用。结合 PyQt5 界面与高性能推理（ONNX Runtime / PyTorch），为皮带输送带提供实时锚杆（bolt）检测、异常报警、历史记录与可交付报告，方便质检与巡检人员在生产线上部署使用。

亮点
- 支持最多 8 路同时独立检测画面（图片 / 视频 / 摄像头）。
- 同时兼容 ONNX 与 PyTorch (.onnx / .pt / .pth) 权重，便于开发与部署。
- 单帧推理由全局信号量序列化，避免并发抢占导致的设备冲突与性能抖动。
- 内置图像增强（CLAHE、亮度/对比度调节）以提高弱光或复杂背景下的召回率。
- 实时报警横幅、统计面板、历史浏览与导出（CSV / JSON / 交互式 HTML 报告）。
- 线程安全的后处理与记录保存，便于批量生成报告与归档。

一眼看懂（快速概览）
- GUI 入口：`main.py` — 构建并管理 8 路检测通道、设备选择、阈值控制与导出功能。
- 模型层：`model_interface.py` — 抽象检测器（BaseDetector） + ONNX / PyTorch 实现 + 工厂函数 `create_detector`。
- 推理工具：`yolov5_utils.py` — letterbox、NMS、坐标映射等 YOLOv5 必备工具，来源于 ultralytics/yolov5 并作了裁剪与适配。
- 视频/图像预处理：`video_processor.py` — 每路的帧预处理、增强与帧率控制。
- 结果管理：`post_processor.py` + `report_generator.py` + `detection_browser.py` + `statistics_panel.py` — 聚合、导出、浏览与可视化。
- 资源目录：`weights/`（放置权重文件与 class_names.txt）、`results/`、`logs/`、`datasets/` 等。

主要功能（更细）
- 多源输入：本地图片、视频文件、摄像头（可同时管理 8 路）。
- 模型灵活：自动识别权重类型（.onnx / .pt/.pth），支持 GPU/CPU 切换（会回退到 CPU）。
- 实时参数调节：置信度（confidence）与 IoU 阈值可即时生效，实时影响下一帧推理。
- 报警策略：可过滤某些类别（如 large_sized_coal）不触发报警；报警横幅包含时间戳与统计信息。
- 导出能力：CSV、JSON、TXT、以及交互式 HTML 报告（可分页/过滤）。
- 历史管理：线程安全的检测记录存储与浏览器界面。

运行要求
- Python 3.8+
- 推荐依赖（最小可运行组）：
  - PyQt5
  - opencv-python
  - numpy
  - onnxruntime (若使用 GPU，请安装对应的 onnxruntime-gpu)
- 若要加载 PyTorch `.pt/.pth` 权重，需安装：
  - torch, torchvision

快速开始（推荐流程）
1. 克隆项目并进入目录：
   ```bash
   git clone https://github.com/kiyobee-152/yolov5-pyqt5.git
   cd yolov5-pyqt5
   ```
2. 建议使用虚拟环境：
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   ```
3. 安装必要依赖：
   ```bash
   pip install -U pip
   pip install opencv-python numpy PyQt5 onnxruntime
   # 如果使用 PyTorch 权重 (.pt/.pth)
   pip install torch torchvision
   ```
4. 将你的模型放入 `weights/` 文件夹：
   - 支持：`*.onnx`, `*.pt`, `*.pth`
   - 在同文件夹放入 `class_names.txt`（每行一个类名，索引按行号从 0 开始；示例仓库已包含）
5. 启动 GUI：
   ```bash
   python main.py
   ```
6. 在左侧选择模型与设备，设置置信度/IoU，点击每路画面的 “图片 / 视频 / 摄像头” 开始检测。使用“导出报告”生成 HTML 报告。

模型 & 类别（说明）
- 若 weights 旁未找到 `class_names.txt`，程序将使用默认类别列表（仓库实现默认为 `['Other_garbage', 'bolt', 'large_sized_coal']`）。
- `class_names.txt` 示例：
  ```
  Other_garbage
  bolt
  large_sized_coal
  ```
- 请确保类别文件与模型训练时使用的类别顺序一致，否则类别映射会错误。

部署与性能建议
- 优先使用 ONNX（onnxruntime），在多数部署场景中 ONNX Runtime 在 CPU / GPU 上更容易获得稳定性能。
- 如果要在 GPU 上运行，请安装 onnxruntime-gpu 并在 GUI 中选择 “GPU”。若环境不支持 GPU，程序会自动回退到 CPU 并提示。
- 性能优化建议：
  - 使用 ONNX 模型的导出版本并做简化（onnx-simplifier / onnxruntime 的优化工具）。
  - 对实时推理节点考虑半精度或量化（动态/静态量化），但请验证精度影响。
  - 调整 FrameRateController 的目标帧率以匹配推理速度，避免丢帧或延迟累积。
- 在资源受限（CPU-only）环境，可降低 GUI 刷新频率或仅在关键帧进行完整推理以节省计算。

开发者注意事项
- 代码结构清晰：`model_interface.py` 为模型抽象，新增模型格式时只需在工厂 `create_detector` 中添加实现即可。
- 推理并发：为了避免多个线程同时占用 GPU，主程序使用了一个全局 Semaphore 来串行化推理请求（如果你要支持真正并行且安全的多 GPU 推理，需要在 Detector 层做更细粒度的设备分配）。
- 日志：`logger.py` 提供统一的日志封装，运行时会在 `logs/` 下保留日志文件。
- 配置：`config.json` 自动保存 UI 设置（置信度、IoU、所选模型、每路图像增强开关等）。

常见问题（FAQ）
- Q: 启动时提示找不到 GPU / CUDA？
  - A: 请确认已安装相应的 GPU 驱动、CUDA 以及与之匹配的 onnxruntime-gpu 或 torch + CUDA 支持；否则选择 CPU 或用 ONNX CPU 进行部署。
- Q: `.pt` 模型加载很慢或出错？
  - A: 仓库的 PyTorch 加载通过 torch.hub 调用 ultralytics/yolov5；离线环境可将官方 yolov5 克隆到项目根目录的 `./yolov5` 或设置 `YOLOV5_REPO` 指向本地仓库。
- Q: 报告导出为空？
  - A: 请确认已有检测历史（在界面产生了若干检测记录），否则导出功能会提示无检测记录可导出。

安全与许可
- 本项目当前未在仓库中声明明确的开源许可证（README 中已有提示）。在公开分发、商用或嵌入到产品之前，请补充适当的许可证并确认第三方模型（如 ultralytics）使用条款。

致谢
- 本项目在预处理/后处理环节借鉴并集成了 ultralytics/YOLOv5 的实现思想（letterbox、NMS、scale_coords 等），并在此基础上封装为工业级的可视化运维工具。
