import torch
import cv2
import numpy as np
from PIL import Image
import os

from zsd import zsd_detector, visualization
from utils import *

from tqdm import tqdm

EVAL_PATH = "./test-all/"
save_dir = "./evaluation_results/"

def main():
    # 初始化
    detector_model, clip_model, clip_preprocess, text, COLORS, description, classes = init()

    # 存储每个类的准确率
    ACCURACY = []

    test_categories = os.listdir(EVAL_PATH)
    for test_category in test_categories:
        # 计算准确率
        acc = 0
        class_root = os.path.join(EVAL_PATH, test_category)

        # 目录下的图像
        image_paths = os.listdir(class_root)
        

        for image_path_ in image_paths:
            save_path = os.path.join(save_dir, test_category)
            mkdir(save_path)
            image_path = os.path.join(class_root, image_path_)
            # 零样本目标检测
            detected_boxes, ids, scores = zsd_detector(detector_model, clip_model, clip_preprocess, text, image_path, threshold=threshold)
            # 判断是否检测到对应类
            for class_id, score in zip(ids, scores):
                label = classes[class_id]
                if label == test_category:
                    if score > eval_threshold:
                        acc += 1
                        break
            
            # 可视化结果
            image_vis = visualization(image_path, detected_boxes, ids, scores, COLORS, classes)
            cv2.imwrite(os.path.join(save_path, image_path_), image_vis)

        # 计算准确率
        print("Class {} accuracy: {}, image number {}.".format(test_category, acc/len(image_paths), len(image_paths)))
        ACCURACY.append(acc/len(image_paths))
        with open(os.path.join(save_dir, "results.txt"),'a') as f: # 'a'表示append,即在原来文件内容后继续写数据（不清楚原有数据）
            f.write("Class {} accuracy: {}, image number {}.\n".format(test_category, acc/len(image_paths), len(image_paths)))
    with open(os.path.join(save_dir, "results.txt"),'a') as f: # 'a'表示append,即在原来文件内容后继续写数据（不清楚原有数据）
        f.write("Average ACC {}".format(sum(ACCURACY)/len(ACCURACY)))
    print("Average ACC {}".format(sum(ACCURACY)/len(ACCURACY)))

main()