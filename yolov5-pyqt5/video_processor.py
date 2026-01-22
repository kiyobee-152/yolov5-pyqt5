# -*- coding: utf-8 -*-
"""
视频预处理模块
提供视频流的预处理功能：分辨率调整、帧率控制、图像增强等
"""
import cv2
import numpy as np
from typing import Tuple, Optional, Callable


class VideoProcessor:
    """视频处理器类"""
    
    def __init__(self, 
                 target_size: Optional[Tuple[int, int]] = None,
                 target_fps: Optional[int] = None,
                 enable_enhancement: bool = False,
                 brightness: float = 1.0,
                 contrast: float = 1.0,
                 saturation: float = 1.0):
        """
        初始化视频处理器
        
        Args:
            target_size: 目标分辨率 (width, height)，None表示保持原分辨率
            target_fps: 目标帧率，None表示保持原帧率
            enable_enhancement: 是否启用图像增强
            brightness: 亮度调整系数 (0.0-2.0)
            contrast: 对比度调整系数 (0.0-2.0)
            saturation: 饱和度调整系数 (0.0-2.0)
        """
        self.target_size = target_size
        self.target_fps = target_fps
        self.enable_enhancement = enable_enhancement
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        处理单帧图像
        
        Args:
            frame: 输入帧
            
        Returns:
            处理后的帧
        """
        # 调整分辨率
        if self.target_size:
            frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)
        
        # 图像增强
        if self.enable_enhancement:
            frame = self._enhance_image(frame)
        
        return frame
    
    def _enhance_image(self, frame: np.ndarray) -> np.ndarray:
        """
        图像增强处理
        
        Args:
            frame: 输入图像
            
        Returns:
            增强后的图像
        """
        # 亮度调整
        if self.brightness != 1.0:
            frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=int((self.brightness - 1.0) * 50))
        
        # 对比度调整
        if self.contrast != 1.0:
            frame = cv2.convertScaleAbs(frame, alpha=self.contrast, beta=0)
        
        # 饱和度调整（转换到HSV空间）
        if self.saturation != 1.0:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv = hsv.astype(np.float32)
            hsv[:, :, 1] = hsv[:, :, 1] * self.saturation
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            hsv = hsv.astype(np.uint8)
            frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return frame
    
    def get_video_info(self, video_path: str) -> dict:
        """
        获取视频信息
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            视频信息字典
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {}
        
        info = {
            'fps': int(cap.get(cv2.CAP_PROP_FPS)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': 0  # 秒
        }
        
        if info['fps'] > 0:
            info['duration'] = info['frame_count'] / info['fps']
        
        cap.release()
        return info
    
    def create_video_writer(self, output_path: str, fps: int, size: Tuple[int, int], 
                           codec: str = 'mp4v') -> cv2.VideoWriter:
        """
        创建视频写入器
        
        Args:
            output_path: 输出视频路径
            fps: 帧率
            size: 视频尺寸 (width, height)
            codec: 编码格式
            
        Returns:
            VideoWriter对象
        """
        fourcc = cv2.VideoWriter_fourcc(*codec)
        return cv2.VideoWriter(output_path, fourcc, fps, size)
    
    def set_enhancement_params(self, brightness: float = None, 
                               contrast: float = None, 
                               saturation: float = None):
        """设置增强参数"""
        if brightness is not None:
            self.brightness = max(0.0, min(2.0, brightness))
        if contrast is not None:
            self.contrast = max(0.0, min(2.0, contrast))
        if saturation is not None:
            self.saturation = max(0.0, min(2.0, saturation))


class FrameRateController:
    """帧率控制器"""
    
    def __init__(self, target_fps: int = 30):
        """
        初始化帧率控制器
        
        Args:
            target_fps: 目标帧率
        """
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps if target_fps > 0 else 0
        self.last_time = None
        import time
        self.time_module = time
    
    def wait_if_needed(self):
        """如果需要，等待以达到目标帧率"""
        if self.frame_time <= 0:
            return
        
        current_time = self.time_module.time()
        if self.last_time is not None:
            elapsed = current_time - self.last_time
            if elapsed < self.frame_time:
                self.time_module.sleep(self.frame_time - elapsed)
        
        self.last_time = self.time_module.time()
    
    def set_fps(self, fps: int):
        """设置目标帧率"""
        self.target_fps = fps
        self.frame_time = 1.0 / fps if fps > 0 else 0
