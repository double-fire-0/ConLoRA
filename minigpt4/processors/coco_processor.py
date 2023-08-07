import torch
from minigpt4.common.registry import registry
from minigpt4.processors.base_processor import BaseProcessor
from omegaconf import OmegaConf
from minigpt4.processors.utils import coco_transforms as T
import numpy as np


class CoCoImageBaseProcessor(BaseProcessor):
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        self.mean = mean
        self.std = std


@registry.register_processor("coco_image_train")
class CoCoImageTrainProcessor(CoCoImageBaseProcessor):

    def __init__(self, mean=None, std=None, patch_image_size=512, max_image_size=512, num_bin=1000, use_oject_random_crop=False):
        super().__init__(mean=mean, std=std)
        self.num_bin = num_bin
        self.use_oject_random_crop = use_oject_random_crop

        if self.use_oject_random_crop:
            self.oject_random_crop = T.ObjectRandmoCrop()
        

        self.image_box_transform = T.Compose([
            T.RandomResize([patch_image_size], max_size=patch_image_size),
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std, max_image_size=max_image_size)
        ])

    def __call__(self, item):
        # item: {'image': Image, "region_coord": [x0, y0, x1, y1]}
        res = {}
        image = item['image']
        w, h = image.size
        boxes_target = {"boxes": [], "labels": [],
                        "area": [], "size": torch.tensor([h, w])}
        x0, y0, x1, y1 = item['region_coord']
        boxes_target["boxes"] = torch.tensor(
            [[float(x0), float(y0), float(x1), float(y1)]])
        boxes_target["labels"] = np.array([0])
        boxes_target["area"] = torch.tensor(
            [(float(x1) - float(x0)) * (float(y1) - float(y0))])

        if self.use_oject_random_crop:
            image, boxes_target = self.oject_random_crop(image, boxes_target)

        patch_image, patch_boxes = self.image_box_transform(
            image, boxes_target)

        quant_x0 = "<bin_{}>".format(
            int((patch_boxes["boxes"][0][0] * (self.num_bin - 1)).round()))
        quant_y0 = "<bin_{}>".format(
            int((patch_boxes["boxes"][0][1] * (self.num_bin - 1)).round()))
        quant_x1 = "<bin_{}>".format(
            int((patch_boxes["boxes"][0][2] * (self.num_bin - 1)).round()))
        quant_y1 = "<bin_{}>".format(
            int((patch_boxes["boxes"][0][3] * (self.num_bin - 1)).round()))
        bin_boxes = "{} {} {} {}".format(
            quant_x0, quant_y0, quant_x1, quant_y1)

        res['patch_image'] = patch_image
        res['patch_boxes'] = patch_boxes
        res['bin_boxes'] = bin_boxes

        return res

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        patch_image_size = cfg.get("patch_image_size", 512)
        max_image_size = cfg.get("max_image_size", 512)
        num_bin = cfg.get('num_bin', 1000)
        use_oject_random_crop = cfg.get('use_oject_random_crop', False)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        return cls(
            mean=mean,
            std=std,
            patch_image_size=patch_image_size,
            max_image_size=max_image_size,
            num_bin=num_bin,
            use_oject_random_crop=use_oject_random_crop,
        )

