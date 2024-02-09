
"""
for generate IR Photo inference report

IR data folder tree:
VST_IR_Photo
├─GC0308
│  ├─Bottle_10cm
│  ....
└─VHS015
    ├─Bottle_10cm
    ....

Usage:

IMAGE:
    1. Inference Video source from Image Folder
        * python tflite_int8_infer.py -s C:\workspace\dataset\data_STD_1_005_001\images --weights ''
        * with "--store_image_result", the result images will be store at C:\workspace\dataset\data_STD_1_005_001\images\{model_name}_infer-result
        * with "--show_cls_conf", the result images will show all class confidence

Video playe:
s       key     : PLAY/ PAUSE
Space   key     : Single Frame Play
c       key     : Capture Frame Image (Original frame)
"""

from cProfile import label
import os
import sys
import copy
import time
import datetime
import argparse

sys.path.append('..')
from pathlib import Path
import pickle
import numpy as np
import cv2
import torch
import torchvision

import math
import pandas as pd
import matplotlib.pyplot as plt

from models.experimental import attempt_load
import math

# import tensorflow as tf

try:  # https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ImportError:
    import tensorflow as tf
    Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate,


# Common parameters
MEDIA_TYPE                  = -1
MEDIA_SOURCE                = r"D:\work\temp_dir\SA_result\image_test"
WEIGHT_FILE                 = r"D:\work\data\recall\yolov5s_320x1_b64_gc_22_1111_2022-12-19_mosaic072\top_n_models\avg\robot_y5s_1317_p-0.974_r-0.9815_map50-0.8278_192x192_ch1-int8.tflite"


# IMAGE SOURCE PARAMETERS
ENABLE_DISPLAY              = False
ENABLE_STORE_IMAGE_RESULT   = False
ENABLE_STORE_VIDEO_RESULT   = False
ENABLE_SHOW_CLASS_CONFIDENCE = False
ENABLE_GENERATE_CONF_EXCEL = False
ENABLE_FILTER_FAR_OBJECT = False
ENABLE_INFERENCE_BACKGROUND = False
ENABLE_CHECK_CONFIDENCE_THRESHOLD = False
IMAGE_SCALE                 = 4

IS_GET_MISMATCH = False
APPEND_MISMATCH_STRING = False
IS_GET_YOLO_LABEL = False
IS_DRAW_LABEL = False

INFERENCE_3CHANNEL = False
USE_SEGMENTATION_MODEL = False

# VIDEO SORUCE PARAMETERS
DEFAULT_WAIT_TIME = 30  # ms


## set each class threshold
# ROT_13.001.002
classNameConf = {
              "Person"       : 0.4
            , "Background"  : 0.4
        }
classConfName = {v: k for k, v in classNameConf.items()}


classNameIndex = {
             "Person"       : 0
           , "Background"    : 1
       }
classIndexName = {v: k for k, v in classNameIndex.items()}

t = time.localtime()
timing = time.strftime("%m_%d_%H_%M", t)


def load_pytorch_weight(file_path, device):

    # device = torch.device(device)
    # input_size = [250, 250] # [h, w]
    model = attempt_load(file_path, device=device) # load FP32 model

    model.float()
    model.eval()
    return model

#copy from utils.metrics
def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

# copy from yolov5.utils.general
def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def Distance_Calculate(y1, x0, x1, imageH, imageW):
    Camera_height = 5.7
    Camera_angle = 0
    Camera_focus = 2.9


    a = math.atan((y1 - imageH/2) / (Camera_focus*1))
    d = Camera_height * abs(math.atan(((90-Camera_angle)*math.pi)/ 180.0 - a))
    d = d * (1 + ((abs((x0 + x1)/2.0 - (imageW / 2)))/200.0))

    return d


def if_filter_far_obj(box_output, imageH, imageW):
    if_filter = False
    
    for i in range(box_output.shape[0]):
        cls_idx = int(box_output[i][-1])
        cls_name = classIndexName[cls_idx]
        for j in range(4):
            if box_output[i][j] <= 0:
                box_output[i][j] = 0
        # print(bndboxes[i])
        x0 = int(box_output[i][0] * imageW)
        y0 = int(box_output[i][1] * imageH)
        x1 = int(box_output[i][2] * imageW)
        y1 = int(box_output[i][3] * imageH)
        x0 = 0 if x0 < 0 else x0
        y0 = 0 if y0 < 0 else y0
        x1 = imageW if x1 > imageW else x1
        y1 = imageH if y1 > imageH else y1
        
        # distance = Distance_Calculate(y1, x0, x1, imageH, imageW)
        # if distance > 30:
        #     if_filter = True
        #     break
        bottom_y_ratio = 0.542 # for GC0308 IR CAM (size:320X240)
        bottom_y = y1 / imageH
        # if bottom_y <= 0.625: # for PC IR CAM
        if bottom_y <= bottom_y_ratio: 
            if_filter = True
            print("too far")
            break
        
    return if_filter

# copy from yolov5.utils.general
def non_max_suppression(prediction,
                        conf_thres=0.25,
                        iou_thres=0.45,
                        classes=None,
                        agnostic=False,
                        multi_label=False,
                        labels=(),
                        max_det=300):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping bounding boxes

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates


    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.1 + 0.03 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * bs
    output_class_conf = [torch.zeros((0, nc), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])
        class_conf = x

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)

            class_conf = torch.cat((conf, x[:, 5:]),1)[conf.view(-1) > conf_thres] # (obj_conf, cls_conf)
            class_conf = class_conf[:, 1:] # (cls_conf)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres] # (xyxy, conf, cls)
            

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]


        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores

        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        output_class_conf[xi] = class_conf[i]
        if (time.time() - t) > time_limit:
            # LOGGER.warning(f'WARNING: NMS time limit {time_limit:.3f}s exceeded')
            print(f'WARNING: NMS time limit {time_limit:.3f}s exceeded')
            break  # time limit exceeded

    return output, output_class_conf

# copy from yolov5.utils.general
def non_max_suppression_objcls(prediction,
                        conf_thres=0.25,
                        iou_thres=0.45,
                        classes=None,
                        agnostic=False,
                        multi_label=False,
                        labels=(),
                        max_det=300,
                        final_confThres = 0.0,
                        nm=0):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping bounding boxes

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    if USE_SEGMENTATION_MODEL:
        nm=32  # number of masks
    upd_tabel = np.zeros([11,12], dtype = int)

    if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output
    device = prediction.device
    mps = 'mps' in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()

    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = (torch.max(prediction[...,4:5]*prediction[...,5:],dim=2).values) > conf_thres  # candidates


    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.1 + 0.03 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    mi = 5 + nc  # mask start index
    output = [torch.zeros((0, 6), device=prediction.device)] * bs
    output_class_conf = [torch.zeros((0, nc), device=prediction.device)] * bs
    output_max_objness = float(0)

    output_candidate = [torch.zeros((0, 6), device=prediction.device)] * bs
    output_candidate_class_conf = [torch.zeros((0, nc), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)


        #print("---------start-----------")
        #print("class_PRs:", x[:, 5:])
        #print("obj_PRs:", x[:, 4:5])

        # If none remain process next image
        if not x.shape[0]:
            if ENABLE_GENERATE_CONF_EXCEL:
                upd_tabel[0,11] += 1
            continue
        
        max_boxes = []
        max_boxes_class_conf = []

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = class_PRs * obj_PRs
        #print("conf:", x[:, 5:])
        #print("-----------end------------")

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])
        class_conf = x

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:mi].max(1, keepdim=True)

            class_conf = torch.cat((conf, x[:, 5:mi]),1)[conf.view(-1) > conf_thres] # (obj_conf, cls_conf)
            class_conf = class_conf[:, 1:] # (cls_conf)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres] # (xyxy, conf, cls)
            

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]


        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            if ENABLE_GENERATE_CONF_EXCEL:
                upd_tabel[0,11] += 1
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores

        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy
        ## output candidates
        output_candidate[xi] = x[i]
        output_candidate_class_conf[xi] = class_conf[i]

        ## pick the max conf output_candidate to be the output
        if len(output_candidate[0].detach().numpy())>=1:
            candidates = output_candidate[0].detach().numpy()
            candidates_class_conf = output_candidate_class_conf[0].detach().numpy()
            for c in range(len(candidates)):
                if candidates[c][4] > final_confThres:
                    if candidates[c][4] > output_max_objness:
                        max_boxes = candidates[c].tolist()
                        max_boxes_class_conf = candidates_class_conf[c].tolist()
                        output_max_objness = candidates[c][4]
        if len(max_boxes) != 0:
            output = [torch.Tensor([max_boxes])]
            output_class_conf = [torch.Tensor([max_boxes_class_conf])]

        if ENABLE_GENERATE_CONF_EXCEL:
            for output_conf in output[0].detach().numpy():
                    output_class_id = int(output_conf[5])
                    output_conf_value = float(output_conf[4])
                    output_conf_value = (math.floor(output_conf_value*10)/10.0)
                    if str(output_conf_value).split('.')[0] == '0':
                        conf_id = int((str(output_conf_value).split('.')[-1]))
                        # print(i, conf_id)
                        upd_tabel[output_class_id,conf_id] += 1
                    elif str(output_conf_value).split('.')[0] == '1':
                        upd_tabel[output_class_id,10] += 1
            if output[0].detach().numpy().shape[0] == 0:
                upd_tabel[0,11] += 1
        #print("upd_tabel", upd_tabel)
        #print("output:", output)
        #print("output_class_conf: ", output_class_conf)
        if (time.time() - t) > time_limit:
            # LOGGER.warning(f'WARNING: NMS time limit {time_limit:.3f}s exceeded')
            print(f'WARNING: NMS time limit {time_limit:.3f}s exceeded')
            break  # time limit exceeded
    
    return output, output_class_conf, upd_tabel

def mt_NMS(pred, conf_thres=0.25, iou_thres=0.45):
    upd_tabel = np.zeros([5,12], dtype = int)

    bs = pred.shape[0]  # batch size
    nc = pred.shape[2] - 5  # number of classes
    xc_indices = pred[..., 4] > conf_thres  # candidates

    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    output = [torch.zeros((0, 6), device=pred.device)] * bs
    output_class_conf = [torch.zeros((0, nc), device=pred.device)] * bs
    for xi, x in enumerate(pred):  # image index, image inference
        x = x[xc_indices[xi]]
        # print("x.shape[0]: ", x.shape[0])
        boxes = xywh2xyxy(x[:, :4])
        if not x.shape[0] :
            continue
        nms_boxes = []
        nms_boxes_class_conf = []
        merged_index = []
        for i, box in enumerate(boxes):
            # print(":::::::::::start:::::::::::::")
            # print("i:", i)
            # print("box:", box)
            # print("merged_index", merged_index)
            if i in merged_index:
                continue
            box = torch.unsqueeze(box, dim=0)
            
            iou = box_iou(box, boxes)
            conf = x[:, 4].clone()
            # print("iou:", iou)
            # print("conf 1 :", conf)
            conf[iou.view(-1) < iou_thres] = 0
            # print("conf 2 :", conf)
            matched_indices = torch.nonzero(conf).view(-1).tolist()
            merged_index.extend(matched_indices)
            score, max_idx = conf.max(0)
            # print("score: ", score, "max_idx:", max_idx)
            class_conf_max, j = x[max_idx, 5:].max(0)
            # print("class_conf_max: ",class_conf_max.unsqueeze(0), j.unsqueeze(0))
            # print("boxes[max_idx]:", boxes[max_idx])
            # print("x[max_idx, 4:5]: ", x[max_idx, 4:5])
            test = torch.cat( (boxes[max_idx], x[max_idx, 4:5], j.unsqueeze(0).float()), 0)
            # print("test 1:", test)
            test = test.cpu().detach().numpy()
            # print("test 2:",test)

            class_conf  = x[max_idx, 5:]
            class_conf = class_conf.cpu().detach().numpy()
            # print("class_conf:", class_conf)
            nms_boxes.append(test.tolist())
            # nms_boxes.append( test )
            nms_boxes_class_conf.append(class_conf.tolist())
            output = nms_boxes # (xyxy, conf, cls)
            output_class_conf = nms_boxes_class_conf
            # print(":::::::::::end:::::::::::::")
        # print("output_class_conf:", output_class_conf)
        # print("output:", output)
        output = [torch.Tensor(output)]
        output_class_conf = [torch.Tensor(output_class_conf)]
    # output = [torch.from_numpy(item).float() for item in output]
    # output_class_conf = [torch.from_numpy(item).float() for item in output_class_conf]
    print("-----------")
    print("output_class_conf:", output_class_conf)
    print("output:", output)
    print("-----------")

    return output, output_class_conf


class TFLiteModel:
    def __init__(self, weight_file: str) -> None:
        self.interpreter = Interpreter(model_path=weight_file)  # load TFLite model
        self.interpreter.allocate_tensors()  # allocate
        self.input_details = self.interpreter.get_input_details()  # inputs
        self.output_details = self.interpreter.get_output_details()  # outputs

    def getInputSize(self):
        [n, inputH, inputW, c] = model.input_details[0]["shape"]
        return (inputH, inputW)

    def infer(self, im):
        input, output = model.input_details[0], model.output_details[0]

        scale, zero_point = input['quantization']
        if input['dtype'] == np.int8:
            im = (im / scale + zero_point).astype(np.int8)  # de-scale
        elif input['dtype'] == np.uint8:
            im = (im / scale + zero_point).astype(np.uint8)  # de-scale

        model.interpreter.set_tensor(input['index'], im)
        model.interpreter.invoke()
        y = model.interpreter.get_tensor(output['index'])
        if input['dtype'] == np.int8 or input['dtype'] == np.uint8:
            scale, zero_point = output['quantization']
            y = (y.astype(np.float32) - zero_point) * scale  # re-scale
        
        return y



def get_image_path_dir(dir_path: str):
    extensions = ('jpeg', 'bmp','jpg','png')
    images = []
    folders = []
    if os.path.isdir(dir_path):
        for r, d, f in os.walk(dir_path):
            [images.append(str(Path(r, i))) for i in f if i.lower().endswith(extensions)]
    return images
    # return [os.path.join(dir_path, f) for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)) and f.endswith(extensions)]

def get_image_path_file(file_path: str):
    image_file_list = []
    with open(file_path, 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            line = line.rstrip()
            image_file_list.append(line)

    return image_file_list

def get_video_path_dir(dir_path: str):
    extensions = ('.mp4', '.MP4')
    return [os.path.join(dir_path, f) for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)) and f.endswith(extensions)]

def infer_video(model:TFLiteModel, media_source:str, media_type, confThr: float=0.4):

    wait_time = 0
    FRAME_COUNT = 5
    SPACE_KEY_ORD = 32
    capture_serial_number = 0
    if media_type == "VIDEO_DEVICE":
        video_file_list = [int(media_source)]
        capture_name = f"device_{media_source}"
        store_path = Path(__file__).parent.resolve() / "capture_device_images"
    elif media_type == "VIDEO_FILE":
        capture_name = str(Path(media_source).stem)
        video_file_list = [media_source]
        if ENABLE_STORE_VIDEO_RESULT:
            model_name = str(Path(WEIGHT_FILE).stem) + "_infer-video"
            store_path = Path(media_source).parents[0] / "infer_video" / model_name
        else:
            store_path = Path(media_source).parents[0] / "capture_video_images"
    elif media_type == "VIDEO_FOLDER":
        video_file_list = get_video_path_dir(media_source)
        model_name = str(Path(WEIGHT_FILE).stem) + "_infer-video"
        store_path = Path(media_source) / "infer_video" / model_name
        cap_store_path = Path(media_source).parents[0]/ "cap_images" / Path(media_source).stem
    total_videos = len(video_file_list)
    if total_videos == 0:
        raise Exception(f"can't get video file: {media_source}")
    print(f"Total Videos: {total_videos}")
    

    

    for video_file in video_file_list:
        if media_type == "VIDEO_FOLDER":
            capture_name = str(Path(video_file).stem)
        if ENABLE_STORE_VIDEO_RESULT:
            store_path.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')


            video_name = f"{capture_name}_infer.mp4"
            if ENABLE_SHOW_CLASS_CONFIDENCE:
                video_name = f"{capture_name}_infer_all_cls.mp4"
            print("video_name: ", video_name)
            video_path = store_path / video_name
            vout = cv2.VideoWriter(str(video_path), fourcc, 20.0, (320, 240), True)

        if media_type == "VIDEO_DEVICE":
            vcap = cv2.VideoCapture( video_file )
        else:
            test_path = Path(video_file)
            vcap = cv2.VideoCapture( str(test_path) )
        if vcap is None:
            print(f"vcap is None")
        if vcap.isOpened() == False:
            print(f"vcap is not open")
        frame_time = (1.0 / vcap.get(cv2.CAP_PROP_FPS)) * 1000 # millisecond

        if ENABLE_INFERENCE_BACKGROUND:
            y0y1_list = []
            
        while True:
            ret, frame = vcap.read()
            if ret:
                frame = cv2.resize(frame, (320, 240))
                cur_pos_time = vcap.get(cv2.CAP_PROP_POS_MSEC)
                print(f"cur_pos_msec: {cur_pos_time}")
                frame_RGB = frame
                if INFERENCE_3CHANNEL:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frameH, frameW, ch = frame.shape
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   
                    frameH, frameW = frame.shape     
                frame_bak = copy.deepcopy(frame_RGB)
                
                (inputH, inputW) = model.getInputSize()

                input_image = cv2.resize(frame, (inputW, inputH))
                im = input_image.astype(np.float32) / 255
                if INFERENCE_3CHANNEL:
                    im = im[None,...]
                    im = np.ascontiguousarray(im)
                else:
                    im = im[None, None, ...]
                    n, c, h, w = im.shape  # batch, channel, height, width
                    im = im.transpose( (0,2,3,1) )
                
                y = model.infer(im)

                if isinstance(y, np.ndarray):
                    y = torch.tensor(y, device=device)
                # bndboxes, class_confs = non_max_suppression(y, conf_thres=confThr)
                bndboxes, class_confs, _ = non_max_suppression_objcls(y, conf_thres=confThr, iou_thres=0.4)
                bndboxes = bndboxes[0].detach().numpy()
                class_confs = class_confs[0].detach().numpy()

                if ENABLE_FILTER_FAR_OBJECT:
                    if_filter = if_filter_far_obj(bndboxes, inputW, inputH)
                    if if_filter:
                        bs = y.shape[0]  # batch size
                        nc = y.shape[2] - 5  # number of classes
                        bndboxes = ([torch.zeros((0, 6), device=y.device)] * bs)[0].detach().numpy()
                        class_confs = ([torch.zeros((0, nc), device=y.device)] * bs)[0].detach().numpy()

                # ==== all class confidence
                all_conf = []
                for i in range(class_confs.shape[0]):
                    one_conf = []
                    for idx in range(len(class_confs[i])):
                        cls_name = classIndexName[idx]
                        cls_conf = class_confs[i][idx]
                        conf_str = str(cls_name) + ":"+ str(cls_conf) + " "
                        one_conf.append(conf_str)
                    all_conf.append(one_conf)



                # ===== convert frame to RGB
                if not INFERENCE_3CHANNEL:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

                for i in range(bndboxes.shape[0]):
                    cls_idx = int(bndboxes[i][-1])
                    x0 = int(bndboxes[i][0] * frameW)
                    y0 = int(bndboxes[i][1] * frameH)
                    x1 = int(bndboxes[i][2] * frameW)
                    y1 = int(bndboxes[i][3] * frameH)
                    x0 = 0 if x0 < 0 else x0
                    y0 = 0 if y0 < 0 else y0
                    x1 = frameW if x1 > frameW else x1
                    y1 = frameW if y1 > frameW else y1

                    if ENABLE_INFERENCE_BACKGROUND:
                        y0y1_list.append([cls_idx, bndboxes[i][4], int(bndboxes[i][0] * inputW), int(bndboxes[i][1] * inputH), int(bndboxes[i][2] * inputW), int(bndboxes[i][3] * inputH)])

                    cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)
                    cv2.rectangle(frame, (x0, y0), (x0+10, y0+10), (0,0,0), -1, cv2.LINE_AA)  # filled
                    cv2.putText(frame, str(cls_idx), (x0, y0+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    if ENABLE_SHOW_CLASS_CONFIDENCE:
                        for j in range(len(all_conf[i])):
                            cv2.putText(frame, str(all_conf[i][j]), (int((x0+x1)/2), y0+(j+1)*15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

                if ENABLE_STORE_VIDEO_RESULT:

                    vout.write(frame)
                if ENABLE_DISPLAY:
                    cv2.imshow('frame', frame)
                    key = cv2.waitKey(wait_time)
                    if key == 27:
                        break
                    elif key == SPACE_KEY_ORD:  # Single frame play
                        wait_time = 0
                    elif key == ord('s'):       # toogle Play / Pause
                        wait_time = DEFAULT_WAIT_TIME if wait_time == 0 else 0
                    elif key == ord('a'):       # backward
                        new_pos_time = cur_pos_time - (frame_time * FRAME_COUNT) 
                        vcap.set(cv2.CAP_PROP_POS_MSEC, new_pos_time)
                        wait_time = 0
                    elif key == ord('d'):       # fast forward
                        new_pos_time = cur_pos_time + (frame_time * FRAME_COUNT) 
                        vcap.set(cv2.CAP_PROP_POS_MSEC, new_pos_time)
                        wait_time = 0
                    elif key == ord('c'):       # Capture frame image
                        store_path.mkdir(parents=True, exist_ok=True)
                        file_name = f"{capture_name}_{int(cur_pos_time)}_{capture_serial_number:04d}.jpeg"
                        print("file_name: ",file_name)
                        file_path = store_path / file_name
                        print("file_path: ",file_path)
                        cv2.imwrite(str(file_path), frame_bak)
                        capture_serial_number += 1
                if ENABLE_INFERENCE_BACKGROUND:
                    # ===== auto capture images
                    # if len(bndboxes) != 0:
                    cap_store_path.mkdir(parents=True, exist_ok=True)
                    # print(cap_store_path)
                    file_name = f"{capture_name}_{int(cur_pos_time)}_{capture_serial_number:04d}.jpeg"
                    # print("file_name: ",file_name)
                    file_path = cap_store_path / Path(video_file).parents[0].name / file_name
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    # print("file_path: ",file_path)
                    cv2.imwrite(str(file_path), frame_bak)

                    # save box image
                    box_file_name = f"{capture_name}_{int(cur_pos_time)}_{capture_serial_number:04d}_box.jpeg"
                    box_file_path = cap_store_path / Path(video_file).parents[0].name / box_file_name
                    box_file_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(box_file_path), frame)  
                    capture_serial_number += 1
            else:
                break
        if ENABLE_INFERENCE_BACKGROUND:
            print(str(store_path / f"{str(Path(video_file).stem)}_bottom_y.txt"))
            with open(str(store_path / f"{str(Path(video_file).stem)}_bottom_y.txt"), 'w') as yf:
                for line in y0y1_list:
                    print(line, file = yf)  
        vcap.release()
        vout.release()


def get_yolo_label(image_file:str):
    """
    return list of [c, xc, yc, w, h]
    """
    label_file = image_file.rsplit('.', 1)[0]
    label_file += '.txt'
    labels = []
    if os.path.exists(label_file):
        with open(label_file, 'r') as fp:
            for line in fp.readlines():
                line = line.rstrip()
                e = line.split(' ')
                label = [int(e[0]), float(e[1]), float(e[2]), float(e[3]), float(e[4])]
                labels.append(label)
                # print(f"labels: {labels[-1]}")
    else:
        print(f"{label_file} does not exist")
    return labels

def xcycwh2xyxy(labels, imw, imh):
    """
    labels: list of [c, xc, yc, w, h]
    return: list of [c, x0, y0, x1, y1]
    """
    new_labels = []
    for l in labels:
        xc = int(l[1] * imw)
        yc = int(l[2] * imh)
        bw = int(l[3] * imw)
        bh = int(l[4] * imh)
        x0 = xc - int(bw/2)
        y0 = yc - int(bh/2)
        x1 = xc + int(bw/2)
        y1 = yc + int(bh/2)

        x0 = 0 if x0 < 0 else x0
        y0 = 0 if y0 < 0 else y0
        x1 = imw if x1 > imw else x1
        y1 = imh if y1 > imh else y1
        new_labels.append( [l[0], x0, y0, x1, y1] )
    return new_labels

    

def draw_label(target_image, labels, imw, imh, draw_class_name=False):
    labels = xcycwh2xyxy(labels, imw, imh)
    for label in labels:
        # elems = label.split(' ')
        cls_idx = label[0]
        cls_name = classIndexName[cls_idx]
        x0 = label[1]
        y0 = label[2]
        x1 = label[3]
        y1 = label[4]

        cv2.rectangle(target_image, (x0, y0), (x1, y1), (0, 0, 255), 2)
        # cv2.rectangle(target_image, (x0, y0), (x0+10, y0+10), (0,0,0), -1, cv2.LINE_AA)  # filled
        if draw_class_name:
            cv2.putText(target_image, cls_name, (x0, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(target_image, str(cls_idx), (x0, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

def box_iou_1(b1, b2):
    """
    b1, b2: [x0, y0, x1, y1]
    """
    def box_area(b):
        if b[2] < b[0] or b[3] < b[1]:
            return 0
        return (b[2] - b[0]) * (b[3] - b[1])
    area1 = box_area(b1)
    area2 = box_area(b2)
    x1 = min(b1[2], b2[2])
    y1 = min(b1[3], b2[3])
    x0 = max(b1[0], b2[0])
    y0 = max(b1[1], b2[1])
    inter = box_area([x0, y0, x1, y1])
    return inter / (area1 + area2 - inter)

def mismatch_iou(labels, predictions, iou_thres=0.4):
    iou_mismatch_count = 0
    cls_mismatch_count = 0
    false_pred = False
    for l in labels:
        iou_match = False
        cls_match = False
        # print(f"label: {l}")
        for p in predictions:
            # print(f"pred: {p}")
            iou = box_iou_1(l[1:], p[1:])
            # print(f"iou: {iou}")
            if iou >= iou_thres:
                iou_match = True
                if l[0] == p[0]:
                    cls_match = True
                    predictions.remove(p)
                    break
        if iou_match == False:
            # print(f"iou not match: {l}")
            iou_mismatch_count += 1 
        else:
            if cls_match == False:
                # print(f"cls not match: {l}")
                cls_mismatch_count += 1
    if len(predictions) > 0:
        # print(f"False Pred: {len(predictions)}")
        # print(f"{predictions}")
        false_pred = True
    
    result = ''
    if iou_mismatch_count > 0:
        result += "iou_not_match_"
    if cls_mismatch_count > 0:
            result += "cls_not_match_"
    if false_pred:
        result += "false_pred_"
    
    return result[:-1]


def convert_to_model_input(image, grayscale = False):
    #image = cv2.resize( image, (model_in_width, model_in_height) )
    if grayscale:
        # image = image[..., None]
        image = image.transpose( (2, 0, 1) )
    else:
        image = np.transpose(image, (2,0,1))
    image = np.ascontiguousarray(image)
    image = np.expand_dims(image, axis=0)
    image = torch.from_numpy(image).float()/255

    return image


def infer_images(model, media_source, media_type, confThr: float=0.4, iou_thres:float=0.4, store_path="",cls="0"):

    if media_type == "IMAGE_FOLDER":
        image_file_list = get_image_path_dir(media_source)
    elif media_type == "IMAGE_FILE_LIST":
        image_file_list = get_image_path_file(media_source)
    elif media_type == "IMAGE_FILE":
        image_file_list = [media_source]
    else:
        image_file_list = []

    total_images = len(image_file_list)
    if total_images == 0:
        raise Exception(f"can't get image file: {media_source}")
    print(f"Total Images: {total_images}")

    # store_path = ""
    if ENABLE_STORE_IMAGE_RESULT and store_path == "":
        #print("CF",str(confThr))
        CF = str.split(str(confThr),'.')
        #list_cf_0 = str.split(str(cf_0),'.')
        #list_cf_1 = str.split(str(cf_1),'.')
        #print("Path(WEIGHT_FILE)",Path(WEIGHT_FILE))
        now = datetime.datetime.now()
        if ("avg" in WEIGHT_FILE) | ("precision" in WEIGHT_FILE) | ("recall" in WEIGHT_FILE):
            Name_list = str(Path(WEIGHT_FILE)).split('/')
            Name_0 = Name_list[-4]
            Name_1 = Name_0.split('_data_config_2023-')[0]
            Name_2 = Name_0.split('_data_config_2023-')[1]
            model_name = Name_1+Name_2+"_"+CF[0]+CF[-1]+"_2023_"+str(now.month)+"_"+str(now.day)+"_STD_Result"
        else:
            Name_list = str(Path(WEIGHT_FILE)).split('/')
            #print("Name_list",Name_list)
            Name_3 = Name_list[-3]
            model_name = Name_3+"_"+CF[0]+CF[-1]+"_2023_"+str(now.month)+"_"+str(now.day)+"_STD_Result"
        
        if not os.path.exists("output/log_output"):
            os.makedirs("output/log_output",exist_ok=True)
        path = 'output/log_output/python_output_%s.txt' %model_name
        inf_log_file = open(path, 'a')

        if media_type == "IMAGE_FOLDER":
            #store_path = Path(media_source).parents[0] / model_name
            name_of_path = str(Path(image_file_list[0]).parents[0]).split('images')
            #print("name_of_path",name_of_path)
            store_path = Path("output/image_output/"+name_of_path[-1])
            store_path_wrong_imgs = Path("output/image_output/"+name_of_path[-1]+"/%s/wrong_imgs"%model_name)
            store_path_bingo_imgs = Path("output/image_output/"+name_of_path[-1]+"/%s/bingo_imgs"%model_name)
            store_path_none = Path("output/image_output/"+name_of_path[-1]+"/none")

        else:
            store_path = Path(image_file_list[0]).parents[2] / model_name
        print(f"STore Infer Result: {store_path}")
        store_path.mkdir(parents=True, exist_ok=True)
        store_path_wrong_imgs.mkdir(parents=True, exist_ok=True)
        store_path_bingo_imgs.mkdir(parents=True, exist_ok=True)
        #store_path_none.mkdir(parents=True, exist_ok=True)


        
        
    ENABLE_GENERATE_CONF_EXCEL = False



    cur_idx = 0
    while (True):
        if cur_idx >= total_images:
            break
        image_file = image_file_list[cur_idx]
        
        
        ## for SJCAM data (HCI testing data)
        if ENABLE_GENERATE_CONF_EXCEL:
            image_name = Path(image_file).name
            image_folder_name = Path(image_file).parents[0].name
            d = image_folder_name[:-1]
            obj_case = int(classNameIndex[str(d)])      

            image_id = obj_case

            upd_tab_id = image_id
        

        if INFERENCE_3CHANNEL:
            image = cv2.imread(image_file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #RGB
        else:
            image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise Exception(f"{image_file} is not an image file")
        if INFERENCE_3CHANNEL:
            imageH, imageW, ch = image.shape # HWC
        else:
            imageH, imageW = image.shape
        enlarge_image = cv2.resize(image, (imageW*IMAGE_SCALE, imageH*IMAGE_SCALE))
        
        if WEIGHT_FILE.endswith(".tflite"):
            (inputH, inputW) = model.getInputSize()
        elif WEIGHT_FILE.endswith(".pt"):
            inputW = imgsz
            inputH = imgsz
        
        # print(f"input hw: [{inputH}, {inputW}]")
        input_image = cv2.resize(image, (inputW, inputH)) # resize RGB
        im = input_image.astype(np.float32) / 255 # HWC, RGB scale 0-255 to 0-1
        if INFERENCE_3CHANNEL:
            im = im[None,...] # batch, channel, height, width            
        else:
            im = im[None, None, ...]
        im = np.ascontiguousarray(im)
        n, c, h, w = im.shape  # batch, channel, height, width
        im = im.transpose( (0,2,3,1) ) # BCHW to BHWC
        if WEIGHT_FILE.endswith(".tflite"):
            y = model.infer(im) # do quantization
        elif WEIGHT_FILE.endswith(".pt"):
            if INFERENCE_3CHANNEL:
                x = convert_to_model_input(image)
            else:
                x = convert_to_model_input(image, True)
            y = model(x)
        
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, device=device)
        # print(y)
        # bndboxes, class_confs = mt_NMS(pred=y, conf_thres=confThr, iou_thres=iou_thres)
        # bndboxes, class_confs = non_max_suppression(y, conf_thres=confThr, iou_thres=iou_thres)
        bndboxes, class_confs, upd_tabel = non_max_suppression_objcls(y, conf_thres=confThr, iou_thres=iou_thres)
        # print("++++++++++++++++infer_images+++++++++++")
        # print("class_confs 1 :", class_confs)
        # print("bndboxes 1 :", bndboxes)
        bndboxes = bndboxes[0].detach().numpy()
        class_confs = class_confs[0].detach().numpy()
        # print("bndboxes 2:", bndboxes)
        # print("class_confs 2 :", class_confs)
        if ENABLE_FILTER_FAR_OBJECT:
            if_filter = if_filter_far_obj(bndboxes, imageH, imageW)
            if if_filter:
                bs = y.shape[0]  # batch size
                nc = y.shape[2] - 5  # number of classes
                bndboxes = ([torch.zeros((0, 6), device=y.device)] * bs)[0].detach().numpy()
                class_confs = ([torch.zeros((0, nc), device=y.device)] * bs)[0].detach().numpy()


        # ==== all class confidence
        all_conf = []
        for i in range(class_confs.shape[0]):
            one_conf = []
            for idx in range(len(class_confs[i])):
                cls_name = classIndexName[idx]
                cls_conf = class_confs[i][idx]
                conf_str = str(cls_name) + ":"+ str(cls_conf) + " "
                one_conf.append(conf_str)
            all_conf.append(one_conf)
        # print("all_conf:", all_conf)

        # ===== convert image to RGB
        if not INFERENCE_3CHANNEL:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            enlarge_image = cv2.cvtColor(enlarge_image, cv2.COLOR_GRAY2BGR)


        # ===== Ground Truth Label
        
        labels = []
        if IS_GET_YOLO_LABEL:
            labels = get_yolo_label(image_file)
        
        if IS_DRAW_LABEL and IS_GET_YOLO_LABEL and len(labels) > 0:
            draw_label(image, labels, imageW, imageH)
            draw_label(enlarge_image, labels, imageW*IMAGE_SCALE, imageH*IMAGE_SCALE, draw_class_name=False)


        # ===== Predition
        r = int(imageH/90)
        predictions = []
        if_pass = False
        for i in range(bndboxes.shape[0]):
            cls_idx = int(bndboxes[i][-1])
            cls_name = classIndexName[cls_idx]
            for j in range(4):
                if bndboxes[i][j] <= 0:
                    bndboxes[i][j] = 0
            # print(bndboxes[i])
            x0 = int(bndboxes[i][0] * imageW)
            y0 = int(bndboxes[i][1] * imageH)
            x1 = int(bndboxes[i][2] * imageW)
            y1 = int(bndboxes[i][3] * imageH)
            x0 = 0 if x0 < 0 else x0
            y0 = 0 if y0 < 0 else y0
            x1 = imageW if x1 > imageW else x1
            y1 = imageH if y1 > imageH else y1

            predictions.append( [cls_idx, x0, y0, x1, y1] )

            if str(cls_idx)=="0":
                cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 255), 20*r)
            else:
                cv2.rectangle(image, (x0, y0), (x1, y1), (255, 0, 0), 4*r)
            cv2.rectangle(image, (x0, y0), (x0+int(imageW/4), y0+int(imageH/4)), (0,0,0), -1, cv2.LINE_AA)  # filled
            cv2.putText(image, str(cls_idx), (x0, y0+int(imageH/8)), cv2.FONT_HERSHEY_SIMPLEX, 1.5*r, (255, 255, 255), 3*r, cv2.LINE_AA)

            x0 = int(bndboxes[i][0] * imageW * IMAGE_SCALE)
            y0 = int(bndboxes[i][1] * imageH * IMAGE_SCALE)
            x1 = int(bndboxes[i][2] * imageW * IMAGE_SCALE)
            y1 = int(bndboxes[i][3] * imageH * IMAGE_SCALE)
            x0 = 0 if x0 < 0 else x0
            y0 = 0 if y0 < 0 else y0
            x1 = imageW*IMAGE_SCALE if x1 > imageW*IMAGE_SCALE else x1
            y1 = imageH*IMAGE_SCALE if y1 > imageH*IMAGE_SCALE else y1
            if str(cls_idx)=="0":
                cv2.rectangle(enlarge_image, (x0, y0), (x1, y1), (0, 255, 255), 20*r)
            else:
                cv2.rectangle(enlarge_image, (x0, y0), (x1, y1), (255, 0, 0), 4*r)
            cv2.rectangle(image, (x0, y0), (x0+int(imageW/4), y0+int(imageH/4)), (0,0,0), -1, cv2.LINE_AA)  # filled
            cv2.putText(image, str(cls_idx), (x0, y0+int(imageH/8)), cv2.FONT_HERSHEY_SIMPLEX, 1.5*r, (255, 255, 255), 3*r, cv2.LINE_AA)
            if ENABLE_SHOW_CLASS_CONFIDENCE:
                for j in range(len(all_conf[i])):
                    if str(cls_idx)=="0":
                        cv2.putText(enlarge_image, str(all_conf[i][j]), (int((x0+x1)/4), int(y0+(j+2)*(imageH/3))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
                    else:
                        cv2.putText(enlarge_image, str(all_conf[i][j]), (int((x0+x1)/4), int(y0+(j+2)*(imageH/3))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
            if ENABLE_CHECK_CONFIDENCE_THRESHOLD:
                ## check confidence threshold
                conf_thres = classNameConf[classIndexName[cls_idx]]
                # if image_id == cls_idx:
                if bndboxes[i][4] > conf_thres:
                    if_pass = True
            if str(cls_idx)==cls:
                action = "Right"
            else:
                action = "Wrong"

                
    
        
        if IS_GET_MISMATCH and store_path != "":
            new_labels = xcycwh2xyxy(labels, imageW, imageH)
            result = mismatch_iou(new_labels, predictions, iou_thres=0.4)
            # print(f"result: {result}")
            if result != "":
                if "iou_not_match" in result:
                    folder_name = "iou_not_match"
                elif "cls_not_match" in result:
                    folder_name = "cls_not_match"
                elif "false_pred" in result:
                    folder_name = "false_pred"
                else:
                    folder_name = "mismatch"

                model_name = str(Path(WEIGHT_FILE).stem)
                if isinstance(store_path, Path) == False:
                    store_path = Path(store_path)
                mismatch_path =  store_path/ model_name
                mismatch_path = mismatch_path / folder_name
                mismatch_path.mkdir(parents=True, exist_ok=True)
                file_name = Path(image_file).stem
                if APPEND_MISMATCH_STRING:
                    file_path = str(mismatch_path.absolute() / file_name) + f"_{result}.jpg"
                else:
                    file_path = str(mismatch_path.absolute() / Path(image_file).name )
                print(f"mismatch: {file_path}")
                cv2.imwrite(file_path, enlarge_image)




        if ENABLE_STORE_IMAGE_RESULT:
            # if len(bndboxes) != 0:
            if isinstance(store_path, Path) == False:
                store_path = Path(store_path)
            if ENABLE_INFERENCE_BACKGROUND:
                file_name = f"{Path(image_file).parents[1].name}_{Path(image_file).parents[0].name}_{Path(image_file).stem}"
            elif ENABLE_CHECK_CONFIDENCE_THRESHOLD:
                file_name = f"{Path(image_file).parents[1].name}_{Path(image_file).parents[0].name}_{Path(image_file).stem}"
            else:
                file_name = Path(image_file).stem
            
            file_name_with_detail = image_file.replace('/','-')
            #print(file_name_with_detail)

            if ENABLE_SHOW_CLASS_CONFIDENCE:
                if ENABLE_CHECK_CONFIDENCE_THRESHOLD:
                    if if_pass:
                        file_path = str(store_path / file_name) + "_l.jpg"
                    else:
                        file_path = str(store_path / file_name) + "_Fail_l.jpg"
                else:
                    #file_path_show_cls = str(store_path / file_name) + "_show_%s_%s_%s_%s_%s.jpg" %(E,T,C,action[1:],cls_prd)
                    #file_path_show_cls_wrong_imgs = str(store_path_wrong_imgs / file_name) + ".jpg"
                    file_path_show_cls_bingo_imgs = str(store_path_bingo_imgs / file_name) + ".jpg"
                    file_path_show_cls_wrong_imgs = str(store_path_wrong_imgs / file_name) + ".jpg"
                    #file_path_show_cls_wrong_imgs = str(store_path_wrong_imgs / file_name_with_detail)
                    #file_path = str(store_path / file_name) + "_l.jpg"
                if action!= 'Right':
                    cv2.imwrite(file_path_show_cls_wrong_imgs, enlarge_image)
                else:
                    cv2.imwrite(file_path_show_cls_bingo_imgs, enlarge_image)

                #cv2.imwrite(file_path, enlarge_image)
            else:
                file_path = str(store_path / file_name) + ".jpg"
                cv2.imwrite(file_path, image)

        if ENABLE_DISPLAY:
            cv2.imshow('image', image)
            cv2.imshow('enlarge image', enlarge_image)
            
            key = cv2.waitKey(0)
            if key == 27:
                break
            elif key == ord('d') or key == 32: # next image
                cur_idx = total_images if cur_idx >= total_images else (cur_idx + 1)
            elif key == ord('a'): # next image
                cur_idx = 0 if (cur_idx - 1) < 0 else (cur_idx - 1)
        else:
            cur_idx = total_images if cur_idx >= total_images else (cur_idx + 1)
    



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device',   default=0, help='0 for cpu, 1 for gpu')
    parser.add_argument('-t', '--type',     type=str, default='image', help='image or video')
    parser.add_argument('-s', '--source',   type=str, default= '/home/vianne/temp/SJCAM_IR_0214_320x240', help='source for image file or folder path of image or video file path or video device id')
    parser.add_argument('-w', '--weight',   type=str, default= '/home/vianne/yolov7/runs/train/y5n_s_2o_320x1_b64_gc_22_1111_2022-11-24/top_n_models/avg/y7_1470_p-0.932_r-0.9396_map50-0.7013-int8.tflite', help='path of tflite weight file')
    parser.add_argument('--store_image_result',     action='store_true', default=True , help="Enable Store image inference result")
    parser.add_argument('--store_video_result',     action='store_true', help="Enable Store video inference result")
    parser.add_argument('--show_cls_conf',          action='store_true', help="Enalbe Show all class confidence")
    parser.add_argument('--display',                action='store_true', help="Enable Display")
    parser.add_argument('--store_path', type=str, help='path for storing result')

    parser.add_argument('--get_mismatch',   action='store_true', help="get prediction and ground truth mismatch, only for image")
    parser.add_argument('--append_mismatch_str',   action='store_true', help="append mismatch string when get mismatch image")
    # parser.add_argument('--draw_label',   action='store_true', help="draw label on image for get mismatch image")
    parser.add_argument('--img_ch', default=1, help="1 or 3 channel")
    parser.add_argument('--img-size', type=int, default=192, help='inference size (pixels)')
    parser.add_argument('--gen_conf_excel', action='store_true', default=False,help="generate object confidence statistics excel")
    parser.add_argument('--conf_thres', type=float, default = 0.4, help="confidence threshold")
    parser.add_argument('--filter_far_obj', action='store_true', default=False, help="filter far objects")
    parser.add_argument('--infer_bg', action='store_true', default=False, help="Enable inference background")
    parser.add_argument('--seg', action='store_true', default=False, help="Inference yolov5-seg model")
    parser.add_argument('--cls',type=str)



    args = parser.parse_args()

    if args.device == 1:
        device = torch.cuda.device(0)
    else:
        device = torch.device("cpu")

    if args.get_mismatch:
        IS_GET_MISMATCH = True
    if args.append_mismatch_str:
        APPEND_MISMATCH_STRING = True
    if IS_GET_MISMATCH:
        IS_GET_YOLO_LABEL = True
        IS_DRAW_LABEL = True

    
    
    MEDIA_TYPE = ""
    MEDIA_SOURCE = args.source
    if args.display:
        ENABLE_DISPLAY = True
    if args.store_image_result:
        ENABLE_STORE_IMAGE_RESULT = True
    if args.store_video_result:
        ENABLE_STORE_VIDEO_RESULT = True
    if args.show_cls_conf:
        ENABLE_SHOW_CLASS_CONFIDENCE = True
    store_path = ""
    if args.store_path:
        store_path = args.store_path

    if args.gen_conf_excel:
        ENABLE_GENERATE_CONF_EXCEL = True
    if args.filter_far_obj:
        ENABLE_FILTER_FAR_OBJECT = True
    if args.infer_bg:
        ENABLE_INFERENCE_BACKGROUND = True

    cls = args.cls

    if args.type == "video":
        if (MEDIA_SOURCE).isdecimal():
            print(f"video device: {MEDIA_SOURCE}")
            MEDIA_TYPE = "VIDEO_DEVICE"
        else:
            if os.path.isfile(MEDIA_SOURCE):
                print(f"video file: {MEDIA_SOURCE}")
                MEDIA_TYPE = "VIDEO_FILE"
            elif os.path.isdir(MEDIA_SOURCE):
                MEDIA_TYPE = "VIDEO_FOLDER"
    elif args.type == "image":
        if os.path.isdir(MEDIA_SOURCE):
            MEDIA_TYPE = "IMAGE_FOLDER"
        elif os.path.isfile(MEDIA_SOURCE):
            if MEDIA_SOURCE.endswith("jpeg") or MEDIA_SOURCE.endswith("jpg") or MEDIA_SOURCE.endswith("png") or MEDIA_SOURCE.endswith("bmp"):
                MEDIA_TYPE = "IMAGE_FILE"
            elif MEDIA_SOURCE.endswith("txt"):
                MEDIA_TYPE = "IMAGE_FILE_LIST"
    if args.img_ch == 3:
        INFERENCE_3CHANNEL = True
    if args.seg:
        USE_SEGMENTATION_MODEL = True
    print("MEDIA_TYPE",MEDIA_TYPE)
    if MEDIA_TYPE == "":
        raise Exception(f"can not locate your source: {args.source}")

    print(f"DEVICE          : {device}")
    print(f"MEDIA_TYPE      : {MEDIA_TYPE}")
    print(f"MEDIA_SOURCE    : {MEDIA_SOURCE}")
    print(f"WEIGHT_FILE     : {WEIGHT_FILE}")
    if (MEDIA_TYPE == "IMAGE_FOLDER"
        or MEDIA_TYPE == "IMAGE_FILE"
        or MEDIA_TYPE == "IMAGE_FILE_LIST"):
        print( "ENABLE DISPLAY   : {}".format("ON" if ENABLE_DISPLAY else "OFF") )
        print( "IMAGE INFER STORE: {}".format("ON" if ENABLE_STORE_IMAGE_RESULT else "OFF"))
        print( "VIDEO INFER STORE: {}".format("ON" if ENABLE_STORE_VIDEO_RESULT else "OFF"))
        print( "SHOW CLASS CONFIDENCE: {}".format("ON" if ENABLE_SHOW_CLASS_CONFIDENCE else "OFF"))
        print( "==================================")
        print( "GET MISMATCH              : {}".format("ON" if IS_GET_MISMATCH else "OFF"))
        print( "APPEND MISMATCH STRING    : {}".format("ON" if APPEND_MISMATCH_STRING else "OFF"))
        print( "GET YOLO LABEL            : {}".format("ON" if IS_GET_YOLO_LABEL else "OFF"))
        print( "GET DRAW LABEL            : {}".format("ON" if IS_DRAW_LABEL else "OFF"))


    if args.weight is not None and os.path.isfile(args.weight):
        WEIGHT_FILE = args.weight
    

    # IS_GET_YOLO_LABEL = True
    # IS_DRAW_LABEL = True
    if WEIGHT_FILE.endswith(".tflite"):
        model = TFLiteModel(WEIGHT_FILE)
    elif WEIGHT_FILE.endswith(".pt"):
        model = load_pytorch_weight(WEIGHT_FILE, device)
        imgsz = args.img_size

    if (MEDIA_TYPE == "VIDEO_DEVICE" 
        or MEDIA_TYPE == "VIDEO_FILE"
        or MEDIA_TYPE == "VIDEO_FOLDER"):
        infer_video(model, MEDIA_SOURCE, MEDIA_TYPE)
    elif (MEDIA_TYPE == "IMAGE_FOLDER"
        or MEDIA_TYPE == "IMAGE_FILE"
        or MEDIA_TYPE == "IMAGE_FILE_LIST"):

        infer_images(model, MEDIA_SOURCE, MEDIA_TYPE, confThr=args.conf_thres, iou_thres=0.4, store_path=store_path, cls=cls)
    else:
        print(f"Undefine MEDIA_SOURCE")
