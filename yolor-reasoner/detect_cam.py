import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

# eigencam imports
import requests
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
import matplotlib.pyplot as plt
# eigencam imports

from utils.google_utils import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from models.models import *
from utils.datasets import *
from utils.general import *


def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

def detect(save_img=False):
    out, ei_path, source, weights, view_img, save_txt, imgsz, cfg, names = \
        opt.output, opt.eigencam, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.cfg, opt.names
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    # model = Darknet(cfg, imgsz).cuda()
    model = Darknet(cfg, imgsz).cpu()
    model.load_state_dict(torch.load(weights, map_location=device)['model'])
    #model = attempt_load(weights, map_location=device)  # load FP32 model
    #imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz, auto_size=64)

    # Get names and colors
    names = load_classes(names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        cam_img = img.copy()
        cam_img = np.float32(cam_img) / 255
        cam_img = np.transpose(cam_img, (1, 2, 0))
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # EigenCAM
        # Path to save CAM images
        # if os.path.exists(ei_path):
        #     shutil.rmtree(ei_path)
        if not os.path.exists(ei_path):
            os.makedirs(ei_path)

        vis_flag = 0
        net_config = cfg.split("/")

        # Choose which layers to use according to config
        scale_array = ['scale1', 'scale2', 'scale3', 'scale4']
        if net_config[-1] == 'yolor_p6.cfg':
            act_index = [214, 220, 226, 232]
        elif net_config[-1] == 'yolor_p6_reasoner.cfg':
            act_index = [216, 224, 232, 240]
        else:
            act_index = []
            print('Undefined net cfg!')

        for i in range(len(act_index)):
            eigencam_layer = [model.module_list[act_index[i]]]
            cam = EigenCAM(model, eigencam_layer, use_cuda=False)
            grayscale_cam = cam(img)[0, :, :]
            cam_image = show_cam_on_image(cam_img, grayscale_cam, use_rgb=False)
            if vis_flag == 1:
                cv2.imshow("window_name", np.asarray(cam_image))
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            inp_img_name = path.split("/")[-1]
            out_img_name = ei_path + "/" + scale_array[i] + "_" + net_config[-1].split(".")[0]  + "_" + inp_img_name
            cv2.imwrite(out_img_name, np.asarray(cam_image))

        xd = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', nargs='+', type=str, default='best_ap50.pt', help='model.pt path(s)')
    parser.add_argument('--weights', nargs='+', type=str, default='last_439.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--eigencam', type=str, default='inference/eigencam', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=1280, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    # parser.add_argument('--cfg', type=str, default='cfg/yolor_p6.cfg', help='*.cfg path')
    parser.add_argument('--cfg', type=str, default='cfg/yolor_p6_reasoner.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.cfg path')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
