# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import multiprocessing as mp
import os

import cv2
from detectron2.utils.visualizer import ColorMode,Visualizer
from detectron2.structures.instances import Instances
import detectron2.data.transforms as T
from detectron2.data import MetadataCatalog
import numpy as np
import torch
from detectron2.checkpoint import DetectionCheckpointer
from fewx.config import get_cfg
# from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import read_image
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger
from GradCAM.head_cam import GradCAM, GradCamPlusPlus
from skimage import io
from torch import nn

# constants
WINDOW_NAME = "COCO detections"

@torch.no_grad()
def Visualize(im, output, train_metadata):
    meta = train_metadata
    # meta.thing_colors = []
    for i in range(len(train_metadata.thing_classes)):
        # meta.thing_colors.append([0,255,0])
        meta.thing_colors[i] = [0,255,0]
    v = Visualizer(im[:, :, ::-1],
                metadata=meta,
                scale=0.3,
                instance_mode=ColorMode.SEGMENTATION
                 )
    all_instances = output["instances"].to("cpu")
        # Top K mode
    draw_instances =  Instances(all_instances.image_size)
    draw_instances.pred_boxes = all_instances.pred_boxes[0:1]
    draw_instances.scores = all_instances.scores[0:1]
    draw_instances.pred_classes = all_instances.pred_classes[0:1]
    out = v.draw_instance_predictions(draw_instances)
    return out.get_image()[:, :, ::-1]

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_last_conv_name(net):
    """
    获取网络的最后一个卷积层的名字
    :param net:
    :return:
    """
    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            layer_name = name
    return layer_name


class GuidedBackPropagation(object):

    def __init__(self, net):
        self.net = net
        for (name, module) in self.net.named_modules():
            if isinstance(module, nn.ReLU):
                module.register_backward_hook(self.backward_hook)
        self.net.eval()

    @classmethod
    def backward_hook(cls, module, grad_in, grad_out):
        """

        :param module:
        :param grad_in: tuple,长度为1
        :param grad_out: tuple,长度为1
        :return: tuple(new_grad_in,)
        """
        return torch.clamp(grad_in[0], min=0.0),

    def __call__(self, inputs, index=0):
        """

        :param inputs: {"image": [C,H,W], "height": height, "width": width}
        :param index: 第几个边框
        :return:
        """
        self.net.zero_grad()
        output = self.net.inference([inputs])
        score = output[0]['instances'].scores[index]
        score.backward()

        return inputs['image'].grad  # [3,H,W]


def norm_image(image):
    """
    标准化图像
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)


def gen_cam(image, mask):
    """
    生成CAM图
    :param image: [H,W,C],原始图像
    :param mask: [H,W],范围0~1
    :return: tuple(cam,heatmap)
    """
    # mask转为heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]  # gbr to rgb

    # 合并heatmap到原始图像
    cam = heatmap + np.float32(image)/255
    return norm_image(cam), heatmap


def gen_gb(grad):
    """
    生guided back propagation 输入图像的梯度
    :param grad: tensor,[3,H,W]
    :return:
    """
    # 标准化
    grad = grad.data.numpy()
    gb = np.transpose(grad, (1, 2, 0))
    return gb

def img_with_box(img, box):
    return



def save_image(image_dicts, input_image_name, network='AirShot-101', output_dir='./vis/grad_head'):
    prefix = os.path.splitext(input_image_name)[0]
    for key, image in image_dicts.items():
        io.imsave(os.path.join(output_dir, '{}-{}-{}-head_vis.jpg'.format(prefix, network, key)), image)


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="configs/fsod/R101/test_R_101_C4_1x_coco5.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--input", help="A list of space separated input images")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
             "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Minimum score for instance predictions to be shown",
    )
    return parser


def main(args):
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    print(cfg)
    # 构建模型
    model = build_model(cfg)
    # 加载权重
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    # metadata = MetadataCatalog.get('voc_2012_val')
    #coco
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

    # 加载图像
    path = os.path.expanduser(args.input)
    original_image = read_image(path, format="BGR")
    height, width = original_image.shape[:2]
    transform_gen = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
    )
    image = transform_gen.get_transform(original_image).apply_image(original_image)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1)).requires_grad_(True)

    inputs = {"image": image, "height": height, "width": width}

    # Grad-CAM
    layer_name = 'roi_heads.res5.2.conv3'
    grad_cam = GradCAM(model, layer_name)
    mask, rpn_box, out = grad_cam(inputs, args.index)  # cam mask
    grad_cam.remove_handlers()

    #
    image_dict = {}
    img = original_image[..., ::-1]
    img1 = original_image[..., ::-1].copy()
    image_dict['im_pred_box'] = Visualize(img,out,metadata)
    # image_dict['pred_box'] = img[y1:y2, x1:x2]
    x1, y1, x2, y2 = rpn_box
    image_dict['im_rpn_box'] = cv2.rectangle(img1, (x1,y1), (x2,y2), (255, 0, 0), 3)
    image_dict['RPN_cam'], _ = gen_cam(img[y1:y2, x1:x2], mask)

    # Grad-CAM++
    grad_cam_plus_plus = GradCamPlusPlus(model, layer_name)
    mask_plus_plus = grad_cam_plus_plus(inputs)  # cam mask
    image_dict['RPN_cam++'], _ = gen_cam(img[y1:y2, x1:x2], mask_plus_plus)
    grad_cam_plus_plus.remove_handlers()

    # 获取类别名称
    # meta = MetadataCatalog.get(
    #     cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
    # )
    # label = meta.thing_classes[class_id]

    # print("label:{}".format(label))
    # # GuidedBackPropagation
    # gbp = GuidedBackPropagation(model)
    # inputs['image'].grad.zero_()  # 梯度置零
    # grad = gbp(inputs)
    # print("grad.shape:{}".format(grad.shape))
    # gb = gen_gb(grad)
    # gb = gb[y1:y2, x1:x2]
    # image_dict['gb'] = gb
    # # 生成Guided Grad-CAM
    # cam_gb = gb * mask[..., np.newaxis]
    # image_dict['cam_gb'] = norm_image(cam_gb)

    save_image(image_dict, os.path.basename(path))


if __name__ == "__main__":
    """
    Usage:export KMP_DUPLICATE_LIB_OK=TRUE
    python detection/demo.py --config-file detection/faster_rcnn_R_50_C4.yaml \
      --input ./examples/pic1.jpg \
      --opts MODEL.WEIGHTS ./model_final_b1acc2.pkl MODEL.DEVICE cpu
    """
    mp.set_start_method("spawn", force=True)
    arguments = get_parser().parse_args()
    main(arguments)
