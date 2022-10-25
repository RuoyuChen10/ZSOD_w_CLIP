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
device = "cuda" if torch.cuda.is_available() else "cpu"


def init():
    # 初始化检测器
    detector_model = init_detector(config_file, checkpoint_file, device=device)

    # 初始化零样本CLIP
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

    # 描述零样本的类别
    description = [
        # AH-64
        "The AH-64 Apache has a four-blade main rotor and a four-blade tail rotor.",
        "The AH-64 is powered by two General Electric T700 turboshaft engines with high-mounted exhausts on either side of the fuselage.",
        # ALH
        "HAL Dhruv has a 4-blade hingeless rotor with a tail rotor pylon/swept main vertical stabilizer at the tail.",
        # AN-32
        "The Antonov An-32 is a turboprop twin-engined military transport aircraft.",
        "The Antonov An-32 type features high-lift wings with automatic leading-edge slats, large triple-slotted trailing edge flaps and an enlarged tailplane and a very large increase in power.",

        # AV-8B
        "AV-8B Harrier is a single-engine ground-attack aircraft.",
        "AV-8B with horizontal stabilizers and shoulder-mounted wings featuring prominent anhedral (downward slope).",

        # B-1
        "The Rockwell B-1 Lancer is a supersonic variable-sweep wing, heavy bomber.",

        # B-2
        "B-2 is a “flying wing,” a configuration consisting essentially of a short but very broad wing with no fuselage and tail.",

        # B-52
        "The Boeing B-52 Stratofortress is an American long-range, subsonic, jet-powered strategic bomber.",

        # C-17
        "The McDonnell Douglas/Boeing C-17 Globemaster III is a large military transport aircraft.",
        "The C-17 Globemaster III is a strategic transport aircraft, able to airlift cargo close to a battle area.",


        # CG
        "Ticonderoga-class guided-missile cruisers are multi-role warships.",
        "The Ticonderoga class of guided-missile cruisers is a class of warships.",

        # CV22
        # "The CV-22 is a tiltrotor aircraft that combines the vertical takeoff, hover and vertical landing qualities of a helicopter with the long-range, fuel efficiency and speed characteristics of a turboprop aircraft.",
        "The Osprey is the world's first production tiltrotor aircraft,[99] with one three-bladed proprotor, turboprop engine, and transmission nacelle mounted on each wingtip.",
        "The Bell Boeing V-22 Osprey is an American multi-mission, tiltrotor military aircraft with both vertical takeoff and landing and short takeoff and landing capabilities.",

        # CVN
        "Aircraft carriers are warships that act as airbases for carrier-based aircraft.",

        # Do-228
        "The Dornier 228 is a twin-turboprop STOL utility aircraft",

        # E2
        "E-2 Hawkeye has a round rotating radar dome that is mounted above its fuselage and wings.",
        "E-2 Hawkeye is an American all-weather, carrier-capable tactical airborne early warning (AEW) aircraft.",

        # F-16
        "F-16 Fighting Falcon is a single-engine multirole fighter aircraft.",

        # F-22
        "F-22",
        "The F-22 has clipped diamond-like delta wings blended into the fuselage with four empennage surfaces and leading edge root extensions running to the upper outboard corner of the caret inlets.",
        "F-22 flight control surfaces include leading-edge flaps, flaperons, ailerons, rudders on the canted vertical stabilizers, and all-moving horizontal tails",
        "F-22 has a refueling boom receptacle centered on its spine and retractable tricycle landing gear.",
        # "the ailerons deflect up, flaperons down, and rudders outwards to increase drag.",

        # LCS
        "The littoral combat ship including a flight deck and hangar.",

        # M142
        "The M142 HIMARS is a light multiple rocket launcher.",

        # MIG29
        "Mikoyan MiG-29 is a twin-engine fighter aircraft.",

        # RQ4
        "RQ-4 Global Hawk is a high-altitude, remotely-piloted surveillance aircraft.",

        # S-400
        "There are 4 cylinders on the back of the S-400, which are loaded with missiles.",
        "Russian S-400 Surface-to-Air Missile System",
        
        # "Su-57"
        "The aircraft has a wide blended wing body fuselage with two widely spaced engines and has all-moving horizontal and vertical stabilisers, with the vertical stabilisers canted for stealth; the trapezoid wings have leading edge flaps, ailerons, and flaperons.",
        "The aircraft incorporates thrust vectoring and large leading edge root extensions that shift the aerodynamic center forward",


        # 其他
        "Person",
        "soldier",
    ]
    classes = [
        "AH-64 Apache",
        "AH-64 Apache",
        "HAL Dhruv",
        "Antonov An-32",
        "Antonov An-32",
        "AV-8B Harrier",
        "AV-8B Harrier",
        "B-1 Lancer",
        "B-2 Spirit",
        "B-52 Stratofortress",
        "C-17",
        "C-17",
        "CG/DDG cruisers",
        "CG/DDG cruisers",
        "CV-22 Osprey",
        "CV-22 Osprey",
        "Aircraft carriers",
        "Dornier 228",
        "E-2 Hawkeye",
        "E-2 Hawkeye",
        "F-16",
        "F-22",
        "F-22",
        "F-22",
        "F-22",
        "Littoral combat ship",
        "M142 HIMARS",
        "MiG-29",
        "RQ-4 Global Hawk",
        "S-400",
        "S-400",
        "Su-57",
        "Su-57",
        "Person",
        "Soldier"
    ]
    # print(len(classes), len(description))
    assert len(classes) == len(description)

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
    scores = [] # 每个检测结果的分数
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
        scores.append(probs.max())
        print(probs)
    return detected_boxes, ids, scores

def visualization(image_path, detected_boxes, ids, scores, COLORS, description):
    """可视化框
    """
    img = cv2.imread(image_path)
    if len(scores) == 0:
        return img

    height, width = img.shape[:2]

    for box, class_id, score in zip(detected_boxes, ids, scores):
        x1, y1, x2, y2 = box
        # 画框
        image_tmp = cv2.rectangle(img, (x1,y1), (x2,y2), COLORS[class_id], int(width/112))
        # 标签
        label = description[class_id]
        # 类别标签
        cv2.putText(image_tmp, label+": "+"%.2f"%(score*100)+"%", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS[class_id], 2)
    return image_tmp

def main():
    # 图像路径
    # image_path = "test-all/AH-64_Apache/ah_0.jpg"
    # image_path = "test-all/ALH/alh_12.jpg"
    # image_path = "test-all/An-32/An32_16.jpg"
    # image_path = "test-all/AV-8B/AV8B_10.jpg"
    # image_path = "test-all/B-1/B1_10.jpg"
    # image_path = "test-all/B-2/B2_7.jpg"
    # image_path = "test-all/B-52/b52_240.jpg"
    # image_path = "test-all/C-17/C17_8.jpg"
    # image_path = "test-all/CG/DDG_338.jpg"
    # image_path = "test-all/CV22/CV22_10.jpg"
    # image_path = "test-all/CVN/LHD_353.jpg"
    # image_path = "test-all/Do-228/do228_10.jpg"
    # image_path = "test-all/E2/E2_13.jpg"
    # image_path = "test-all/F-16/f16_85.jpg"
    # image_path = "test-all/F-22/f22_20.jpg"
    # image_path = "test-all/LCS/LCS_223.jpg"
    # image_path = "test-all/M142/M142_8.jpg"
    # image_path = "test-all/MIG29/mig29_15.jpg"
    # image_path = "test-all/RQ4/rq4_24.jpg"
    # image_path = "test-all/S-400/S400_8.jpg"
    image_path = "test-all/Su-57/Su-57.png"

    
    # 初始化
    detector_model, clip_model, clip_preprocess, text, COLORS, description, classes = init()

    # 零样本目标检测
    detected_boxes, ids, scores = zsd_detector(detector_model, clip_model, clip_preprocess, text, image_path)
    
    # 可视化
    image_vis = visualization(image_path, detected_boxes, ids, scores, COLORS, classes)

    cv2.imwrite("result.jpg", image_vis)

main()




