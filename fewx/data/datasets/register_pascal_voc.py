# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import os

from detectron2.data import DatasetCatalog, MetadataCatalog

from detectron2.data.datasets.coco import load_coco_json

__all__ = ["register_voc_instances"]

"""
This file contains functions to register a COCO-format Pascal VOC to the DatasetCatalog.
"""
DATASET_CATEGORIES = [
    {"name": "aeroplane", "id": 1, "isthing": 1, "color": [220, 20, 60]},
    {"name": "bicycle", "id": 2, "isthing": 1, "color": [219, 142, 185]},
    {"name": "bird", "id": 3, "isthing": 1, "color": [220, 120, 60]},
    {"name": "boat", "id": 4, "isthing": 1, "color": [219, 42, 185]},
    {"name": "bottle", "id": 5, "isthing": 1, "color": [20, 20, 60]},
    {"name": "bus", "id": 6, "isthing": 1, "color": [19, 142, 185]},
    {"name": "car", "id": 7, "isthing": 1, "color": [220, 20, 160]},
    {"name": "cat", "id": 8, "isthing": 1, "color": [119, 142, 185]},
    {"name": "chair", "id": 9, "isthing": 1, "color": [220, 220, 60]},
    {"color": [250, 0, 30], "isthing": 1, "id": 10, "name": "cow"},
    {"color": [165, 42, 42], "isthing": 1, "id": 11, "name": "diningtable"},
    {"color": [255, 77, 255], "isthing": 1, "id": 12, "name": "dog"},
    {"color": [0, 226, 252], "isthing": 1, "id": 13, "name": "horse"},
    {"color": [182, 182, 255], "isthing": 1, "id": 14, "name": "motorbike"},
    {"color": [0, 82, 0], "isthing": 1, "id": 15, "name": "person"},
    {"color": [120, 166, 157], "isthing": 1, "id": 16, "name": "pottedplant"},
    {"color": [110, 76, 0], "isthing": 1, "id": 17, "name": "sheep"},
    {"color": [174, 57, 255], "isthing": 1, "id": 18, "name": "sofa"},
    {"color": [199, 100, 0], "isthing": 1, "id": 19, "name": "train"},
    {"color": [72, 0, 118], "isthing": 1, "id": 20, "name": "tvmonitor"},
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

def register_voc_instances(name, json_file, image_root):
    """
    purpose: register dataset to DatasetCatalog,
             register metadata to MetadataCatalog and set attribute
    """
    DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))
    MetadataCatalog.get(name).set(json_file=json_file, 
                                  image_root=image_root, 
                                  evaluator_type="coco", 
                                  **get_dataset_instances_meta())