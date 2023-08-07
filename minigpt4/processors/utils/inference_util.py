import re
import torch
from torchvision.utils import draw_bounding_boxes
import logging
from random import randint
import torch


def get_image_with_bbox(image, output_text, bin_boxes=None):
    try:
        if image.dtype is not torch.uint8:
            image = (image * 255).to(torch.uint8)
        if image.dim != 3:
            image = image[0]
            
        output_box = re.findall(r"<bin_(.+?)>", output_text)
        output_box = [int(out) for out in output_box]

        # fill the boxes with random if not enough
        if len(output_box) < 4:
            _, h, w = image.shape
            output_box_bk = output_box.copy()
            if len(output_box) == 2:
                output_box += [randint(output_box[0], w), randint(output_box[1], h)]
            elif len(output_box) == 3:
                output_box += [randint(output_box[1], h)]
            print(f'use random box from {output_box_bk} to {output_box}')

        if bin_boxes is not None:
            bin_boxes = re.findall(r"<bin_(.+?)>", bin_boxes)
            bin_boxes = [int(out) for out in bin_boxes]

        box = [output_box] if bin_boxes is None else [bin_boxes, output_box]
        box = torch.tensor(box, dtype=torch.int)
        image = draw_bounding_boxes(image, box, width=5, colors=["orange"] if bin_boxes is not None else ["blue", "orange"], fill=False)
        return image  # [Tensor(C,H,W)] uint8 
    except Exception as error:
        # we don't want it to block training
        logging.exception(f"get_image_with_bbox")
        raise error

def unnorm_image(image_norm):
    try:
        has_batch = True
        if len(image_norm.shape) == 3:
            image_norm = image_norm.unsqueeze(0)
            has_batch = False
        device, dtype = image_norm.device, image_norm.dtype
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device, dtype=dtype).view(1, -1, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device, dtype=dtype).view(1, -1, 1, 1)
        image_unnorm = image_norm * std + mean
        # image_unnorm = image_unnorm.to(torch.uint8)
        return image_unnorm if has_batch else image_unnorm[0]
    except Exception as error:
        logging.exception("unnorm_image")
        raise error
