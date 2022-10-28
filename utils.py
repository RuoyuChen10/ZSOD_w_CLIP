import torch
import numpy as np
import os

from mmdet.apis import init_detector, inference_detector
from mmcv import Config
import clip

config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# 从 model zoo 下载 checkpoint 并放在 `checkpoints/` 文件下
# 网址为: http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
device = "cuda" if torch.cuda.is_available() else "cpu"

threshold=0.5
eval_threshold = 0.2

cfg = Config.fromfile(config_file)

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
        "person",
        "soldier",
    ]
    classes = [
        "AH-64_Apache",
        "AH-64_Apache",
        "HAL_Dhruv",
        "An-32",
        "An-32",
        "AV-8B_Harrier",
        "AV-8B_Harrier",
        "B-1_Lancer",
        "B-2_Spirit",
        "B-52_Stratofortress",
        "C-17",
        "C-17",
        "CG_DDG_Cruisers",
        "CG_DDG_Cruisers",
        "CV-22_Osprey",
        "CV-22_Osprey",
        "Aircraft_Carriers",
        "Dornier-228",
        "E-2_Hawkeye",
        "E-2_Hawkeye",
        "F-16",
        "F-22",
        "F-22",
        "F-22",
        "F-22",
        "Littoral_Combat_Ship",
        "M142_HIMARS",
        "MiG-29",
        "RQ-4_Global_Hawk",
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

def mkdir(name):
    '''
    Create folder
    '''
    isExists=os.path.exists(name)
    if not isExists:
        os.makedirs(name)
    return 0