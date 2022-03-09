# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import os

from detectron2.data import DatasetCatalog, MetadataCatalog

from detectron2.data.datasets.coco import load_coco_json

"""
This file contains functions to register a COCO-format SUBT to the DatasetCatalog.
"""
DATASET_CATEGORIES = [
    {"name": "backpac", "id": 1, "isthing": 1, "color": [220, 20, 60]},
    {"name": "rop", "id": 2, "isthing": 1, "color": [219, 142, 185]},
    {"name": "ven", "id": 3, "isthing": 1, "color": [220, 120, 60]},
    {"name": "helme", "id": 4, "isthing": 1, "color": [219, 42, 185]},
    {"name": "dril", "id": 5, "isthing": 1, "color": [20, 20, 60]},
    {"name": "fire extinguishe", "id": 6, "isthing": 1, "color": [19, 142, 185]},
    {"name": "helmet-ligh", "id": 7, "isthing": 1, "color": [220, 20, 160]},
    {"name": "survivo", "id": 8, "isthing": 1, "color": [119, 142, 185]},
    {"name": "cell phon", "id": 9, "isthing": 1, "color": [220, 220, 60]},
]

def get_dataset_instances_meta():
    """
    purpose: get metadata of dataset from DATASET_CATEGORIES
    return: dict[metadata]
    """
    thing_ids = [k["id"] for k in DATASET_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in DATASET_CATEGORIES if k["isthing"] == 1]
    # assert len(thing_ids) == 2, len(thing_ids)
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in DATASET_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret

def register_subt_instances(name, json_file, image_root):
    """
    purpose: register dataset to DatasetCatalog,
             register metadata to MetadataCatalog and set attribute
    """
    DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))
    MetadataCatalog.get(name).set(json_file=json_file, 
                                  image_root=image_root, 
                                  evaluator_type="coco", 
                                  **get_dataset_instances_meta())
#     checkout_dataset_annotation(name, json_file, image_root)
    
# # 查看数据集标注
# def checkout_dataset_annotation(name, json_file, image_root):
#     dataset_dicts = load_coco_json(json_file, image_root, name)
#     import random
#     for d in random.sample(dataset_dicts,3):
#         img = cv2.imread(d["file_name"])
#         visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get(name), scale=1.5)
#         vis = visualizer.draw_dataset_dict(d)
#         cv2.imshow('show', vis.get_image()[:, :, ::-1])
#         cv2.waitKey(0)