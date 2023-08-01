import os
import sys
import re
import six
import time
import math
import random

import PIL
import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset


from minigpt4.datasets.datasets.base_dataset import BaseDataset

__all__ = ['OCRLmdbDataset']

class LmdbDataset(Dataset):
    def __init__(self, root):

        self.root = root
        self.env = lmdb.open(
            root,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        if not self.env:
            print("cannot open lmdb from %s" % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            self.nSamples = int(txn.get("num-samples".encode()))

        #     self.filtered_index_list = []
        #     for index in range(self.nSamples):
        #         index += 1  # lmdb starts with 1
        #         label_key = "label-%09d".encode() % index
        #         label = txn.get(label_key).decode("utf-8")

        #         # length filtering
        #         length_of_label = len(label)
        #         if length_of_label > opt.batch_max_length # 25:
        #             continue

        #         self.filtered_index_list.append(index)

        #     self.nSamples = len(self.filtered_index_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), f"index range error index:{index} > len:{len(self)}"
        index = self.filtered_index_list[index]

        with self.env.begin(write=False) as txn:
            label_key = "label-%09d".encode() % index
            label = txn.get(label_key).decode("utf-8")
            img_key = "image-%09d".encode() % index
            imgbuf = txn.get(img_key)
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)

            try:
                img = PIL.Image.open(buf).convert("RGB")

            except IOError:
                print(f"Corrupted image for {index}")
                # make dummy image and dummy label for corrupted image.
                img = PIL.Image.new("RGB", (self.opt.imgW, self.opt.imgH))
                label = "[dummy_label]"

        return (img, label)


class OCRLmdbDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, lmdb_root=None):
        super().__init__(vis_processor, text_processor)

        self.inner_dataset = LmdbDataset(lmdb_root)
        # self.tensorboard_writer = SummaryWriter(log_dir='/hetu_group/wangyan/tunnel/data_tensorboard')

    def __getitem__(self, index):
        image, ocr_text = self.inner_dataset[index]

        image = self.vis_processor(image)
        ocr_text = self.text_processor(ocr_text)

        return {
            'image': image,
            'text_input': ocr_text,
        }

    def __len__(self):
        return len(self.inner_dataset)

