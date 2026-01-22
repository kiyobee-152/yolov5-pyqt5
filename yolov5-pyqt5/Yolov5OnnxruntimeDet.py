# -*- coding: utf-8 -*-
# Copyright (C) 2024/8/31 FIRC. All Rights Reserved
# @Time    : 2024/8/31 下午9:37
# @Author  : FIRC
# @Email   : 1623863129@qq.com
# @File    : Yolov5OnnxruntimeDet.py
# @Software: PyCharm-2024.2
# @ Function Description:
import cv2
import numpy as np
from yolov5_utils import *
import cv2
import numpy as np
import onnxruntime
import torch


class Yolov5OnnxruntimeDet(object):
    def __init__(self, weights='./weights/yolov5s.onnx', names=None):
        self.names = names
        if names is None:
            self.load_labels('./weights/class_names.txt')
        self.img_size = (640, 640)  # 训练权重的传入尺寸
        cuda = torch.cuda.is_available()
        self.device = 'cuda' if cuda else 'cpu'  # 根据pytorch是否支持gpu选择设备
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else [
            'CPUExecutionProvider']  # 选择onnxruntime
        print('load onnx weights...')
        self.sess = onnxruntime.InferenceSession(weights, providers=providers)  # 加载模型
        self.confidence = 0.45
        self.iou = 0.45
        self.output_name = self.sess.get_outputs()[0].name
        self.input_name = self.sess.get_inputs()[0].name
        #warm up
        self.inference_image(np.zeros((300,300,3), dtype=np.uint8))
        print('weights loaded!')

    def load_labels(self, file):
        with open(file, 'r') as f:
            self.names = f.read().rstrip('\n').split('\n')

    def inference_image(self, image):
        # 预处理
        img = letterbox(image, self.img_size, stride=64, auto=False)[0]
        # 转tensor
        img = img.transpose((2, 0, 1))[::-1]  # HWC 转 CHW，BGR 转 RGB
        # 返回一个连续的array，其内存是连续的。
        img = np.ascontiguousarray(img)
        # 放入设备
        img = torch.from_numpy(img).to(self.device)
        # uint8 转 fp32
        img = img.float()
        # 归一化
        img /= 255
        # 扩大批量调暗
        if len(img.shape):
            img = img[None]

        # 推理
        img = img.cpu().numpy()  # 传入cpu并转成numpy格式
        pred_onnx = torch.tensor(self.sess.run([self.output_name], {self.input_name: img})[0])
        # nms
        pred = non_max_suppression(pred_onnx, self.confidence, self.iou, classes=None, agnostic=False, max_det=1000)

        # 转换
        result_list = []
        for i, det in enumerate(pred):
            if len(det):
                # 将坐标 (xyxy) 从 img_shape 重新缩放为 img0_shape
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()
                for *xyxy, conf, cls in reversed(det):  # 从末尾遍历
                    # 将xyxy合并至一个维度,锚框的左上角和右下角
                    xyxy = (torch.tensor(xyxy).view(1, 4)).view(-1)
                    result_list.append(
                        [self.names[int(cls)], round(float(conf), 2), int(xyxy[0]), int(xyxy[1]), int(xyxy[2]),
                         int(xyxy[3])])

        return result_list
    def draw_image(self, result_list, opencv_img):
        if len(result_list) == 0:
            return opencv_img
        for result in result_list:
            label_text = result[0] + ',' + str(result[1])
            opencv_img = self.__draw_image(opencv_img, [result[2], result[3], result[4], result[5]], label_text)
        return opencv_img

    def __draw_image(self, opencv_img, box, label='', line_width=None, box_color=(255, 0, 0),
                     txt_box_color=(200, 200, 200),
                     txt_color=(255, 255, 255)):
        '''
        code modified yolov5-6.0
        Args:
            opencv_img:
            box: [xmin,ymin,xmax,ymax]
            label: text,not support chinese
            line_width: None
            box_color:
            txt_box_color:
            txt_color:

        Returns:opencv image with draw box and label
        '''
        lw = line_width or max(round(sum(opencv_img.shape) / 2 * 0.003), 2)  # line width
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(opencv_img, p1, p2, box_color, thickness=lw, lineType=cv2.LINE_AA)
        if label:
            tf = max(lw - 1, 1)  # font thickness
            w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
            outside = p1[1] - h - 3 >= 0  # label fits outside box
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(opencv_img, p1, p2, txt_box_color, -1, cv2.LINE_AA)  # filled
            cv2.putText(opencv_img, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, lw / 3, txt_color,
                        thickness=tf, lineType=cv2.LINE_AA)
        return opencv_img


    def imshow(self, result_list, opencv_img):
        if len(result_list) > 0:
            opencv_img = self.draw_image(result_list, opencv_img)
        cv2.imshow('result', opencv_img)
        cv2.waitKey(0)

    def start_camera(self, camera_index=0):
        cap = cv2.VideoCapture(camera_index)
        frame_fps = int(cap.get(cv2.CAP_PROP_FPS))
        # 获取视频帧宽度和高度
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("video fps={},width={},height={}".format(frame_fps, frame_width, frame_height))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            result_list = self.inference_image(frame)
            result_img = self.draw_image(result_list, frame)
            cv2.imshow('frame', result_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def start_video(self, video_file):
        cap = cv2.VideoCapture(video_file)
        frame_fps = int(cap.get(cv2.CAP_PROP_FPS))
        # 获取视频帧宽度和高度
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("video fps={},width={},height={}".format(frame_fps, frame_width, frame_height))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            result_list = self.inference_image(frame)
            result_img = self.draw_image(result_list, frame)
            cv2.imshow('frame', result_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    detector = Yolov5OnnxruntimeDet(weights=r'E:\ai\yolov5-6.0\pre-model\yolov5s.onnx')
    detector.load_labels(r'E:\official-model\yolov8\labels.txt')
    # detector.start_video(r'D:\car.mp4')
    img = cv2.imread(r'E:\test.png')
    result_list = detector.inference_image(img)
    detector.imshow(result_list=result_list, opencv_img=img)
