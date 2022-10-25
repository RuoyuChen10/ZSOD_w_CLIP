import math
import torch

import cv2
import matplotlib.pyplot as plt
import numpy as np

from mmdet.apis import init_detector, inference_detector
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector

from mmcv.parallel import collate, scatter
from mmcv.ops import RoIPool
from mmcv import Config, DictAction
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

import torch
import clip
from PIL import Image

config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# 从 model zoo 下载 checkpoint 并放在 `checkpoints/` 文件下
# 网址为: http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
device = 'cuda:0'


def init():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 初始化检测器
    detector_model = init_detector(config_file, checkpoint_file, device=device)

    # 初始化零样本CLIP
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

    # 描述零样本的类别
    description = [
        # F-22
        "F-22",
        "The F-22 has clipped diamond-like delta wings blended into the fuselage with four empennage surfaces and leading edge root extensions running to the upper outboard corner of the caret inlets.",
        "Flight control surfaces include leading-edge flaps, flaperons, ailerons, rudders on the canted vertical stabilizers, and all-moving horizontal tails",
        "The aircraft has a refueling boom receptacle centered on its spine and retractable tricycle landing gear."
        "the ailerons deflect up, flaperons down, and rudders outwards to increase drag.",
        # "Su-35",
        # Su-57
        # "Su-57",
        "The aircraft has a wide blended wing body fuselage with two widely spaced engines and has all-moving horizontal and vertical stabilisers, with the vertical stabilisers canted for stealth; the trapezoid wings have leading edge flaps, ailerons, and flaperons.",
        "The aircraft incorporates thrust vectoring and large leading edge root extensions that shift the aerodynamic center forward",
    ]
    classes = [
        "F-22",
        "F-22",
        "F-22",
        "F-22",
        # "Su-35",
        # "Su-57",
        "Su-57",
        "Su-57"
    ]
    # 编码文本
    text = clip.tokenize(description).to(device)

    np.random.seed(300)
    COLORS = np.random.uniform(0, 255, size=(len(description), 3))

    return detector_model, clip_model, clip_preprocess, text, COLORS, description, classes

def zsd_detector(detector_model, clip_model, clip_preprocess, text, image_path, threshold=0.5):
    """零样本检测器

    Args:
        detector_model: 目标检测模型
        clip_model: CLIP模型
        clip_preprocess: CLIP预处理
        image_path: 输入图像路径
        threshold: 物体检测阈值

    Return:
        detected_boxes: 模型检测框结果
        ids: 模型每个框检测的类别
    """
    # 读入图像
    image = cv2.imread(image_path)
    # 检测可能的物体
    out = inference_detector(detector_model, image)

    # 保存检测框
    detected_boxes = []
    for i, pred in enumerate(out):
        for *box, score in pred:
            if score < 0.5:
                break
            box = tuple(np.round(box).astype(int).tolist())
            detected_boxes.append(box)

    # 零样本检测
    ids = [] # 每个检测框预测结果
    image = Image.open(image_path)
    for i in range(len(detected_boxes)):
        ## 剪切
        cropped = image.crop(detected_boxes[i])
        ## 预处理
        clip_input = clip_preprocess(cropped).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits_per_image, logits_per_text = clip_model(clip_input, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        ids.append(probs.argmax())

    return detected_boxes, ids

def visualization(image_path, detected_boxes, ids, COLORS, description):
    """可视化框
    """
    img = cv2.imread(image_path)
    height, width = img.shape[:2]

    for box, class_id in zip(detected_boxes, ids):
        x1, y1, x2, y2 = box
        # 画框
        image_tmp = cv2.rectangle(img, (x1,y1), (x2,y2), COLORS[class_id], int(width/112))
        # 标签
        label = description[class_id]
        # 类别标签
        cv2.putText(image_tmp, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS[class_id], 2)
    return image_tmp

def main():
    # 图像路径
    image_path = "images/Su-57.png"
    # 初始化
    detector_model, clip_model, clip_preprocess, text, COLORS, description, classes = init()

    # 零样本目标检测
    detected_boxes, ids = zsd_detector(detector_model, clip_model, clip_preprocess, text, image_path)

    # 可视化
    image_vis = visualization(image_path, detected_boxes, ids, COLORS, classes)

    cv2.imwrite("result.jpg", image_vis)

main()




