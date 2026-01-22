# -*- coding: utf-8 -*-
"""
后处理模块
提供检测结果的后处理功能：保存结果、导出报告、统计信息等
"""
import os
import json
import csv
import cv2
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from collections import defaultdict


class DetectionResult:
    """检测结果数据类"""
    
    def __init__(self, class_name: str, confidence: float, bbox: Tuple[int, int, int, int],
                 timestamp: Optional[str] = None, frame_id: Optional[int] = None):
        """
        初始化检测结果
        
        Args:
            class_name: 类别名称
            confidence: 置信度
            bbox: 边界框 (x1, y1, x2, y2)
            timestamp: 时间戳
            frame_id: 帧ID
        """
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox
        self.timestamp = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.frame_id = frame_id
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'class_name': self.class_name,
            'confidence': float(self.confidence),
            'bbox': list(self.bbox),
            'timestamp': self.timestamp,
            'frame_id': self.frame_id
        }


class PostProcessor:
    """后处理器类"""
    
    def __init__(self, output_dir: str = './results'):
        """
        初始化后处理器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        self.detection_history: List[DetectionResult] = []
        self.statistics: Dict[str, int] = defaultdict(int)
        os.makedirs(output_dir, exist_ok=True)
    
    def add_detection(self, result_list: List[List], frame_id: Optional[int] = None):
        """
        添加检测结果到历史记录
        
        Args:
            result_list: 检测结果列表，格式: [class_name, confidence, x1, y1, x2, y2]
            frame_id: 帧ID
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for result in result_list:
            det_result = DetectionResult(
                class_name=result[0],
                confidence=result[1],
                bbox=(result[2], result[3], result[4], result[5]),
                timestamp=timestamp,
                frame_id=frame_id
            )
            self.detection_history.append(det_result)
            self.statistics[result[0]] += 1
    
    def get_statistics(self) -> Dict[str, int]:
        """获取统计信息"""
        return dict(self.statistics)
    
    def get_detection_summary(self) -> str:
        """获取检测摘要文本"""
        if not self.statistics:
            return "未检测到目标"
        
        summary = "检测统计:\n"
        total = sum(self.statistics.values())
        summary += f"总计: {total}\n"
        for class_name, count in self.statistics.items():
            percentage = (count / total * 100) if total > 0 else 0
            summary += f"{class_name}: {count} ({percentage:.1f}%)\n"
        
        return summary
    
    def save_image(self, image: np.ndarray, filename: Optional[str] = None, 
                   subfolder: str = 'images') -> str:
        """
        保存图像
        
        Args:
            image: 图像数组
            filename: 文件名，None则自动生成
            subfolder: 子文件夹名称
            
        Returns:
            保存的文件路径
        """
        folder = os.path.join(self.output_dir, subfolder)
        os.makedirs(folder, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detection_{timestamp}.jpg"
        
        filepath = os.path.join(folder, filename)
        cv2.imwrite(filepath, image)
        return filepath
    
    def save_video(self, frames: List[np.ndarray], filename: Optional[str] = None,
                   fps: int = 30, subfolder: str = 'videos') -> str:
        """
        保存视频
        
        Args:
            frames: 帧列表
            filename: 文件名
            fps: 帧率
            subfolder: 子文件夹名称
            
        Returns:
            保存的文件路径
        """
        if not frames:
            return ""
        
        folder = os.path.join(self.output_dir, subfolder)
        os.makedirs(folder, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detection_{timestamp}.mp4"
        
        filepath = os.path.join(folder, filename)
        height, width = frames[0].shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
        return filepath
    
    def export_json(self, filename: Optional[str] = None) -> str:
        """
        导出检测结果为JSON格式
        
        Args:
            filename: 文件名
            
        Returns:
            保存的文件路径
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detections_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        data = {
            'statistics': self.statistics,
            'total_detections': len(self.detection_history),
            'detections': [det.to_dict() for det in self.detection_history]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return filepath
    
    def export_csv(self, filename: Optional[str] = None) -> str:
        """
        导出检测结果为CSV格式
        
        Args:
            filename: 文件名
            
        Returns:
            保存的文件路径
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detections_{timestamp}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['时间戳', '帧ID', '类别', '置信度', 'X1', 'Y1', 'X2', 'Y2'])
            
            for det in self.detection_history:
                writer.writerow([
                    det.timestamp,
                    det.frame_id if det.frame_id is not None else '',
                    det.class_name,
                    det.confidence,
                    det.bbox[0], det.bbox[1],
                    det.bbox[2], det.bbox[3]
                ])
        
        return filepath
    
    def export_report(self, filename: Optional[str] = None) -> str:
        """
        导出文本报告
        
        Args:
            filename: 文件名
            
        Returns:
            保存的文件路径
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"report_{timestamp}.txt"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 50 + "\n")
            f.write("皮带传送带锚杆检测报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(self.get_detection_summary() + "\n")
            f.write("-" * 50 + "\n")
            f.write("详细检测记录:\n")
            f.write("-" * 50 + "\n")
            
            for i, det in enumerate(self.detection_history, 1):
                f.write(f"\n检测 #{i}:\n")
                f.write(f"  时间: {det.timestamp}\n")
                if det.frame_id is not None:
                    f.write(f"  帧ID: {det.frame_id}\n")
                f.write(f"  类别: {det.class_name}\n")
                f.write(f"  置信度: {det.confidence:.2f}\n")
                f.write(f"  位置: ({det.bbox[0]}, {det.bbox[1]}) - ({det.bbox[2]}, {det.bbox[3]})\n")
        
        return filepath
    
    def clear_history(self):
        """清空历史记录"""
        self.detection_history.clear()
        self.statistics.clear()
    
    def get_recent_detections(self, count: int = 10) -> List[DetectionResult]:
        """获取最近的检测结果"""
        return self.detection_history[-count:] if count > 0 else self.detection_history
