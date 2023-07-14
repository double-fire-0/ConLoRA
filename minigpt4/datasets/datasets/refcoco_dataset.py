import base64
from io import BytesIO

from minigpt4.datasets.datasets.base_dataset import BaseDataset
from PIL import Image
from minigpt4.datasets.datasets.file_dataset import FileDataset
from torch.utils.tensorboard import SummaryWriter


class RefCoCoDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, tsv_path, tsv_cols=[0, 4, 2, 3], vis_root=None, ann_paths=None):
        super().__init__(vis_processor, text_processor)

        # TODO list dataset input
        assert not isinstance(tsv_path, list)

        self.inner_dataset = FileDataset(tsv_path, tsv_cols)
        # self.tensorboard_writer = SummaryWriter(log_dir='/hetu_group/wangyan/tunnel/data_tensorboard')

    def __getitem__(self, index):
        uniq_id, base64_str, text, region_coord = self.inner_dataset[index]

        image = Image.open(
            BytesIO(base64.urlsafe_b64decode(base64_str))).convert("RGB")
        region_coord = region_coord.strip().split(',')

        vis_items = {'image': image, 'region_coord': region_coord}
        vis_items = self.vis_processor(vis_items)
        # vis_items['patch_image'] = patch_image
        # vis_items['patch_boxes'] = patch_boxes
        # vis_items['bin_boxes'] = bin_boxes
        caption = self.text_processor(text)

        vis_items['description'] = caption

        return {
            'image': vis_items['patch_image'],
            'bin_boxes': vis_items['bin_boxes'],
            'description': vis_items['description'],
        }

    def __len__(self):
        return len(self.inner_dataset)

    # def collater(self, samples):
