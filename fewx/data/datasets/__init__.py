from . import builtin  # ensure the builtin datasets are registered
from .register_coco import register_coco_instances
from .register_subt import register_subt_instances
from .register_pascal_voc import register_voc_instances
__all__ = [k for k in globals().keys() if "builtin" not in k and not k.startswith("_")]
