# -*- coding: utf-8 -*-
"""
通用模型接口模块
支持多种深度学习模型格式（ONNX、PyTorch等）
"""
from abc import ABC, abstractmethod
import numpy as np
import cv2
from typing import List, Tuple, Optional


class BaseDetector(ABC):
    """检测器基类，定义通用接口"""
    
    def __init__(self, weights: str, names: Optional[List[str]] = None, conf_thres: float = 0.45, iou_thres: float = 0.45):
        """
        初始化检测器
        
        Args:
            weights: 模型权重文件路径
            names: 类别名称列表
            conf_thres: 置信度阈值
            iou_thres: IOU阈值
        """
        self.weights = weights
        self.names = names or []
        self.confidence = conf_thres
        self.iou = iou_thres
        self.img_size = (640, 640)  # 默认输入尺寸
        
    @abstractmethod
    def load_model(self):
        """加载模型"""
        pass
    
    @abstractmethod
    def inference_image(self, image: np.ndarray) -> List[List]:
        """
        对单张图像进行推理
        
        Args:
            image: 输入图像 (numpy array, BGR格式)
            
        Returns:
            检测结果列表，每个元素格式: [class_name, confidence, x1, y1, x2, y2]
        """
        pass
    
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple, Tuple]:
        """
        图像预处理（通用）
        
        Args:
            image: 输入图像
            
        Returns:
            处理后的图像, 缩放比例, 填充大小
        """
        from yolov5_utils import letterbox
        return letterbox(image, self.img_size, stride=64, auto=False)
    
    def draw_image(self, result_list: List[List], opencv_img: np.ndarray) -> np.ndarray:
        """
        在图像上绘制检测结果
        
        Args:
            result_list: 检测结果列表
            opencv_img: 原始图像
            
        Returns:
            绘制了检测框的图像
        """
        if len(result_list) == 0:
            return opencv_img
        
        for result in result_list:
            label_text = f"{result[0]}, {result[1]:.2f}"
            opencv_img = self._draw_box(opencv_img, 
                                       [result[2], result[3], result[4], result[5]], 
                                       label_text)
        return opencv_img
    
    def _draw_box(self, img: np.ndarray, box: List[int], label: str = '', 
                  line_width: Optional[int] = None, 
                  box_color: Tuple[int, int, int] = (255, 0, 0),
                  txt_box_color: Tuple[int, int, int] = (200, 200, 200),
                  txt_color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
        """绘制检测框和标签"""
        lw = line_width or max(round(sum(img.shape) / 2 * 0.003), 2)
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(img, p1, p2, box_color, thickness=lw, lineType=cv2.LINE_AA)
        
        if label:
            tf = max(lw - 1, 1)
            w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]
            outside = p1[1] - h - 3 >= 0
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(img, p1, p2, txt_box_color, -1, cv2.LINE_AA)
            cv2.putText(img, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                       0, lw / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)
        return img
    
    def set_confidence(self, conf: float):
        """设置置信度阈值"""
        self.confidence = conf
    
    def set_iou(self, iou: float):
        """设置IOU阈值"""
        self.iou = iou


class YOLOv5ONNXDetector(BaseDetector):
    """YOLOv5 ONNX模型检测器"""
    
    def __init__(self, weights: str, names: Optional[List[str]] = None, conf_thres: float = 0.45, iou_thres: float = 0.45):
        super().__init__(weights, names, conf_thres, iou_thres)
        self.sess = None
        self.input_name = None
        self.output_name = None
        self.device = 'cpu'
        self.load_model()
    
    def load_model(self):
        """加载ONNX模型"""
        import onnxruntime
        import torch
        
        cuda = torch.cuda.is_available()
        self.device = 'cuda' if cuda else 'cpu'
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        
        print(f'正在加载ONNX模型: {self.weights}')
        self.sess = onnxruntime.InferenceSession(self.weights, providers=providers)
        self.output_name = self.sess.get_outputs()[0].name
        self.input_name = self.sess.get_inputs()[0].name
        
        # 加载类别名称
        if not self.names:
            import os
            names_file = os.path.join(os.path.dirname(self.weights), 'class_names.txt')
            if os.path.exists(names_file):
                with open(names_file, 'r', encoding='utf-8') as f:
                    self.names = f.read().rstrip('\n').split('\n')
        
        # Warm up
        dummy_img = np.zeros((300, 300, 3), dtype=np.uint8)
        self.inference_image(dummy_img)
        print('模型加载完成!')
    
    def inference_image(self, image: np.ndarray) -> List[List]:
        """推理单张图像"""
        import torch
        from yolov5_utils import non_max_suppression, scale_coords
        
        # 预处理
        img, ratio, pad = self.preprocess(image)
        img = img.transpose((2, 0, 1))[::-1]  # HWC转CHW，BGR转RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()
        img /= 255
        if len(img.shape) == 3:
            img = img[None]
        
        # 推理
        img_np = img.cpu().numpy()
        pred_onnx = torch.tensor(self.sess.run([self.output_name], {self.input_name: img_np})[0])
        
        # NMS
        pred = non_max_suppression(pred_onnx, self.confidence, self.iou, classes=None, agnostic=False, max_det=1000)
        
        # 转换结果
        result_list = []
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    xyxy = (torch.tensor(xyxy).view(1, 4)).view(-1)
                    cls_name = self.names[int(cls)] if int(cls) < len(self.names) else f"class_{int(cls)}"
                    result_list.append([
                        cls_name,
                        round(float(conf), 2),
                        int(xyxy[0]), int(xyxy[1]),
                        int(xyxy[2]), int(xyxy[3])
                    ])
        
        return result_list


def create_detector(weights: str, model_type: str = 'auto', **kwargs) -> BaseDetector:
    """
    工厂函数：根据模型文件创建对应的检测器
    
    Args:
        weights: 模型权重文件路径
        model_type: 模型类型 ('auto', 'onnx', 'pytorch')
        **kwargs: 其他参数
        
    Returns:
        检测器实例
    """
    import os
    
    if model_type == 'auto':
        # 自动检测模型类型
        ext = os.path.splitext(weights)[1].lower()
        if ext == '.onnx':
            model_type = 'onnx'
        elif ext in ['.pt', '.pth']:
            model_type = 'pytorch'
        else:
            model_type = 'onnx'  # 默认
    
    if model_type == 'onnx':
        return YOLOv5ONNXDetector(weights, **kwargs)
    elif model_type == 'pytorch':
        # 可以扩展支持PyTorch模型
        raise NotImplementedError("PyTorch模型支持待实现")
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
