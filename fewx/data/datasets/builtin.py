import os
from .register_subt import register_subt_instances
from .register_coco import register_coco_instances
from .register_pascal_voc import register_voc_instances
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata

# ==== Predefined datasets and splits for COCO ==========

_PREDEFINED_SPLITS_COCO = {}
_PREDEFINED_SPLITS_COCO["coco"] = {
    "coco_2017_train_nonvoc": ("coco/train2017", "coco/new_annotations/final_split_non_voc_instances_train2017.json"),
    "coco_2017_train_voc_1_shot": ("coco/train2017", "coco/new_annotations/final_split_voc_1_shot_instances_train2017.json"),
    "coco_2017_train_voc_2_shot": ("coco/train2017", "coco/new_annotations/final_split_voc_2_shot_instances_train2017.json"),
    "coco_2017_train_voc_3_shot": ("coco/train2017", "coco/new_annotations/final_split_voc_3_shot_instances_train2017.json"),
    "coco_2017_train_voc_4_shot": ("coco/train2017", "coco/new_annotations/final_split_voc_4_shot_instances_train2017.json"),
    "coco_2017_train_voc_5_shot": ("coco/train2017", "coco/new_annotations/final_split_voc_5_shot_instances_train2017.json"),
    "coco_2017_train_voc_10_shot": ("coco/train2017", "coco/new_annotations/final_split_voc_10_shot_instances_train2017.json"),
    "coco_2014_minivalidation": ("coco/val2014", "coco/annotations2014/instances_minival2014.json"),
}
# ==== Predefined datasets and splits for SUBT ==========
_PREDEFINED_SPLITS_SUBT = {}
_PREDEFINED_SPLITS_SUBT["subt"] = {
    "subt_val_a_1": ("SUBT/JPEGImages", "SUBT/new_annotations/1_shot_a_val.json"),
    "subt_val_a": ("SUBT/JPEGImages", "SUBT/new_annotations/val_a.json"),
    "subt_val_b_1": ("SUBT/JPEGImages", "SUBT/new_annotations/1_shot_b_val.json"),
    "subt_val_b": ("SUBT/JPEGImages", "SUBT/new_annotations/val_b.json"),
    "subt_val_c_1": ("SUBT/JPEGImages", "SUBT/new_annotations/1_shot_c_val.json"),
    "subt_val_c": ("SUBT/JPEGImages", "SUBT/new_annotations/val_c.json"),
    "subt_val_d_1": ("SUBT/JPEGImages", "SUBT/new_annotations/1_shot_d_val.json"),
    "subt_val_d": ("SUBT/JPEGImages", "SUBT/new_annotations/val_d.json"),
    "subt_val_e_1": ("SUBT/JPEGImages", "SUBT/new_annotations/1_shot_e_val.json"),
    "subt_val_e": ("SUBT/JPEGImages", "SUBT/new_annotations/val_e.json"),
    "subt_val_f_1": ("SUBT/JPEGImages", "SUBT/new_annotations/1_shot_f_val.json"),
    "subt_val_f": ("SUBT/JPEGImages", "SUBT/new_annotations/val_f.json"),
    "subt_val_g_1": ("SUBT/JPEGImages", "SUBT/new_annotations/1_shot_g_val.json"),
    "subt_val_g": ("SUBT/JPEGImages", "SUBT/new_annotations/val_g.json"),
    "subt_val_h_1": ("SUBT/JPEGImages", "SUBT/new_annotations/1_shot_h_val.json"),
    "subt_val_h": ("SUBT/JPEGImages", "SUBT/new_annotations/val_h.json"),
    "subt_val_i_1": ("SUBT/JPEGImages", "SUBT/new_annotations/1_shot_i_val.json"),
    "subt_val_i": ("SUBT/JPEGImages", "SUBT/new_annotations/val_i.json"),
}

# ==== Predefined datasets and splits for Pascal VOC ==========

_PREDEFINED_SPLITS_PASCAL = {}
_PREDEFINED_SPLITS_PASCAL["pascal_voc"] = {
    "voc_2012_train_1": ("voc/JPEGImages", "voc/new_annotations/voc2012_1_shot_val.json"),
    "voc_2012_train_2": ("voc/JPEGImages", "voc/new_annotations/voc2012_2_shot_val.json"),
    "voc_2012_train_3": ("voc/JPEGImages", "voc/new_annotations/voc2012_3_shot_val.json"),
    "voc_2012_train_4": ("voc/JPEGImages", "voc/new_annotations/voc2012_4_shot_val.json"),
    "voc_2012_train_5": ("voc/JPEGImages", "voc/new_annotations/voc2012_5_shot_val.json"),
    "voc_2012_train_10": ("voc/JPEGImages", "voc/new_annotations/voc2012_10_shot_val.json"),
    "voc_2012_validation": ("voc/JPEGImages","voc/annotations/pascal_val2012.json"),
}

def register_all_coco(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_COCO.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

def register_all_subt(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_SUBT.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_subt_instances(name=key,
                                    json_file=os.path.join(root, json_file) if "://" not in json_file else json_file,
                                    image_root=os.path.join(root, image_root),)

def register_all_voc(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_PASCAL.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_voc_instances(name=key,
                                    json_file=os.path.join(root, json_file) if "://" not in json_file else json_file,
                                    image_root=os.path.join(root, image_root),)

# Register them all under "./datasets"
_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_coco(_root)
register_all_subt(_root)
register_all_voc(_root)