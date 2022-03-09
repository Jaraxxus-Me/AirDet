category_map = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    4: 'airplane',
    5: 'bus',
    6: 'train',
    7: 'truck',
    8: 'boat',
    9: 'traffic light',
    10: 'fire hydrant',
    11: 'stop sign',
    12: 'parking meter',
    13: 'bench',
    14: 'bird',
    15: 'cat',
    16: 'dog',
    17: 'horse',
    28: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    27: 'backpack',
    28: 'umbrella',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis racket',
    44: 'bottle',
    46: 'wine glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted plant',
    65: 'bed',
    67: 'dining table',
    70: 'toilet',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush'
}


import torch
import json
import os
import cv2
from detectron2.utils.visualizer import ColorMode,Visualizer
from detectron2.structures.instances import Instances

def Visualize(input, output, dir, train_metadata, thresh):
    im = cv2.imread(input['file_name'])
    name = input['file_name'].split("/")[-1]
    v = Visualizer(im[:, :, ::-1],
                metadata=train_metadata,
                scale=0.8
                 )
    all_instances = output["instances"].to("cpu")
    if thresh >= 1:
        # Top K mode
        draw_instances =  Instances(all_instances.image_size)
        draw_instances.pred_boxes = all_instances.pred_boxes[0:thresh+1]
        draw_instances.scores = all_instances.scores[0:thresh+1]
        draw_instances.pred_classes = all_instances.pred_classes[0:thresh+1]
    if thresh < 1:
        # Score Threshhold mode
        draw_instances =  Instances(all_instances.image_size)
        n=0
        for score in all_instances.scores:
            if score >= thresh:
                n+=1
        if n==0:
            n=3
        draw_instances.pred_boxes = all_instances.pred_boxes[0:n+1]
        draw_instances.scores = all_instances.scores[0:n+1]
        draw_instances.pred_classes = all_instances.pred_classes[0:n+1]
    out = v.draw_instance_predictions(draw_instances)
    # boxes = v._convert_boxes(output["instances"].pred_boxes.to('cpu'))
    # for box in boxes:
    #     box = (round(box[0]), round(box[1]), round(box[2]) - round(box[0]), round(box[3] - box[1]))
    #     out = v.draw_text(f"{box[2:4]}", (box[0], box[1]))
    cv2.imwrite(os.path.join(dir,name), out.get_image()[:, :, ::-1])



# all_res_path="./output/finetune_dir/R_50_C4_1x/inference/instances_predictions.pth"
# annos="./datasets/coco/annotations/instances_val2017.json"
# base_img_path = './datasets/coco/val2017'
# ann_f = open(annos, 'r')
# ann_json = json.load(ann_f)
# find = {}
# for anno in ann_json['images']:
#     find[anno['id']] = anno['file_name']
#     find[anno['file_name']] = (anno['height'], anno['width'])

# all_res = torch.load(all_res_path)
# for res in all_res: 
#     idx = res['image_id']
#     img_file = find[idx]
#     img_size = find[img_file]
#     im = cv2.imread(os.path.join(base_img_path,img_file))
#     outputs = res['instances']
#     scores=[]
#     pred_boxes=[]
#     pre_cls=[]
#     for output in outputs:
#         if output["score"]>0.7:
#             scores.append(output["score"])
#             pred_boxes.append(output["bbox"])
#             pre_cls.append(category_map[output["category_id"]])
#     if len(scores):
#         print(str(idx)+':'+img_file)  
#         i = Instances(img_size)
#         i.set('scores',scores)
#         i.set('pred_classes',pre_cls)
#         i.set('pred_boxes',pred_boxes)
#         v = Visualizer(im[:, :, ::-1],
#                     #    metadata=meta_img, 
#                     #instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
#         )
#         out = v.draw_instance_predictions(i)
#         cv2.imwrite('vis/'+img_file, out.get_image()[:, :, ::-1])