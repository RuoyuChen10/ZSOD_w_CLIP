import torch
import cv2
import numpy as np
from PIL import Image

from mmdet.core.post_processing import fast_nms, multiclass_nms

from utils import *

def zsd_detector(detector_model, clip_model, clip_preprocess, text, image_path, threshold=0.3, use_nms = True, score_thr=0.05, iou_threshold=0.4):
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
            if score < threshold:
                break
            box = tuple(np.round(box).astype(int).tolist())
            detected_boxes.append(box)

    # 零样本检测
    ids = [] # 每个检测框预测结果
    max_scores = [] # 每个检测结果的分数
    all_scores = [] # 检测分数one-hot向量
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
        max_scores.append(probs.max())
        # 最后一类为背景类，需要编码进去
        one_hot = np.zeros(len(text)+1)
        one_hot[probs.argmax()] = 1
        one_hot[:-1] = probs[0] * one_hot[:-1]
        all_scores.append(one_hot)

    if use_nms and len(ids) != 0: # 使用NMS去除重复的框，但是相同类物体不同描述不能视为同一物体。
        det_bboxes, det_labels = multiclass_nms(
            torch.Tensor(detected_boxes).to(device), 
            torch.Tensor(all_scores).to(device), 
            score_thr = score_thr, 
            nms_cfg = dict(type='nms', iou_threshold = iou_threshold),
            max_num = 100)
        # 重写检测框等
        detected_boxes = det_bboxes[:,:4].cpu().numpy().astype(int)
        ids = det_labels.cpu().numpy()
        max_scores = det_bboxes[:,4].cpu().numpy()
        
        return detected_boxes, ids, max_scores
    # 此处返回的是无NMS处理的检测框
    return detected_boxes, ids, max_scores

def visualization(image_path, detected_boxes, ids, scores, COLORS, classes):
    """可视化框
    """
    img = cv2.imread(image_path)
    if len(scores) == 0:
        return img
    # 通过图片大小调整框的粗细与字体的大小
    height, width = img.shape[:2]

    for box, class_id, score in zip(detected_boxes, ids, scores):
        x1, y1, x2, y2 = box
        # 画框
        image_tmp = cv2.rectangle(img, (x1,y1), (x2,y2), COLORS[class_id], int(width/112))
        # 标签
        label = classes[class_id]
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
    # image_path = "test-all/Littoral_Combat_Ship/LCS_223.jpg"
    # image_path = "test-all/M142_HIMARS/M142_2.jpg"
    # image_path = "test-all/MiG-29/mig29_15.jpg"
    # image_path = "test-all/RQ-4_Global_Hawk/rq4_24.jpg"
    # image_path = "test-all/S-400/S400_5.jpg"
    # image_path = "test-all/Su-57/Su-57.png"
    
    # image_path = "images/F-22_1.png"
    image_paths = [
        "images/MiG-29.jpg",
        "images/M142_HIMARS.jpg",
        "images/RQ-4_Global_Hawk.jpg",
        "images/S400.jpg"
    ]

    # 初始化
    detector_model, clip_model, clip_preprocess, text, COLORS, description, classes = init()

    for image_path in image_paths:
        # 零样本目标检测
        detected_boxes, ids, scores = zsd_detector(detector_model, clip_model, clip_preprocess, text, image_path, threshold=threshold, use_nms = True)
        
        # 可视化
        image_vis = visualization(image_path, detected_boxes, ids, scores, COLORS, classes)

        height, width = image_vis.shape[:2]
        image_vis = cv2.resize(image_vis, (400, int(400/width*height)), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(image_path.replace("images", "results"), image_vis)

if __name__ == "__main__":
    main()




