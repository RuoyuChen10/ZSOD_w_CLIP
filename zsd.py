import torch
import cv2
import numpy as np
from PIL import Image

from utils import *

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
            if score < threshold:
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
    image_path = "test-all/Littoral_Combat_Ship/LCS_223.jpg"
    # image_path = "test-all/M142_HIMARS/M142_2.jpg"
    # image_path = "test-all/MiG-29/mig29_15.jpg"
    # image_path = "test-all/RQ-4_Global_Hawk/rq4_24.jpg"
    # image_path = "test-all/S-400/S400_5.jpg"
    # image_path = "test-all/Su-57/Su-57.png"

    # 初始化
    detector_model, clip_model, clip_preprocess, text, COLORS, description, classes = init()

    # 零样本目标检测
    detected_boxes, ids, scores = zsd_detector(detector_model, clip_model, clip_preprocess, text, image_path, threshold=threshold)
    
    # 可视化
    image_vis = visualization(image_path, detected_boxes, ids, scores, COLORS, classes)

    cv2.imwrite("result.jpg", image_vis)

if __name__ == "__main__":
    main()




