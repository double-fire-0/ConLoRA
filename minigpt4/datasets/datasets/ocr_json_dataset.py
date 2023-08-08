from minigpt4.datasets.datasets.base_dataset import BaseDataset
from PIL import Image
from minigpt4.datasets.datasets.file_dataset import FileDataset


import json
import os


class OCRJsonDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, json_path, vis_root=None):
        super().__init__(vis_processor, text_processor, vis_root)

        assert not isinstance(json_path, list)

        with open(json_path) as f:
            lines = f.readlines()

        self.num = len(lines)
        self.metas = []
        for line in lines:
            data = json.loads(line)
            data['image_path'] = os.path.join(self.vis_root, data['image_name'])
            self.metas.append(data)

    def __getitem__(self, index):
        data = self.metas[index]
        image_path, text, region_coord = data['image_path'], data['captation'], data['box']


        image = Image.open(image_path).convert('RGB')

        vis_items = {'image': image, 'region_coord': region_coord}
        vis_items = self.vis_processor(vis_items)
        # vis_items['patch_image'] = patch_image
        # vis_items['patch_boxes'] = patch_boxes
        # vis_items['bin_boxes'] = bin_boxes
        caption = self.text_processor(text)

        vis_items['description'] = caption

        return {
            'image': vis_items['patch_image'],
            'input_item': vis_items['bin_boxes'],
            'output_item': vis_items['description'],
        }

    def __len__(self):
        return self.num
