import os
import logging
import warnings

from minigpt4.common.registry import registry
from minigpt4.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from minigpt4.datasets.datasets.laion_dataset import LaionDataset
from minigpt4.datasets.datasets.cc_sbu_dataset import CCSBUDataset, CCSBUAlignDataset
from minigpt4.datasets.datasets.refcoco_dataset import RefCoCoDataset
from minigpt4.datasets.datasets.ocr_json_dataset import OCRJsonDataset


@registry.register_builder("refcoco")
class RefCoCoBuilder(BaseDatasetBuilder):
    train_dataset_cls = RefCoCoDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/refcoco/defaults.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        tsv_path = build_info.tsv_path
        tsv_cols = build_info.tsv_cols

        datasets = dict()

        if not os.path.exists(tsv_path):
            warnings.warn("tsv path {} does not exist.".format(tsv_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            tsv_path=tsv_path,
            tsv_cols=tsv_cols
        )

        return datasets


# TODO fix copy codes
@registry.register_builder("refcocog")
class RefCoCoGBuilder(BaseDatasetBuilder):
    train_dataset_cls = RefCoCoDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/refcocog/defaults.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        tsv_path = build_info.tsv_path
        tsv_cols = build_info.tsv_cols

        datasets = dict()

        if not os.path.exists(tsv_path):
            warnings.warn("tsv path {} does not exist.".format(tsv_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            tsv_path=tsv_path,
            tsv_cols=tsv_cols
        )

        return datasets


# TODO fix copy codes
@registry.register_builder("refcocop")
class RefCoCoPBuilder(BaseDatasetBuilder):
    train_dataset_cls = RefCoCoDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/refcocop/defaults.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        tsv_path = build_info.tsv_path
        tsv_cols = build_info.tsv_cols

        datasets = dict()

        if not os.path.exists(tsv_path):
            warnings.warn("tsv path {} does not exist.".format(tsv_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            tsv_path=tsv_path,
            tsv_cols=tsv_cols
        )

        return datasets


@registry.register_builder("ocr_art")
class OcrARTBuilder(BaseDatasetBuilder):
    train_dataset_cls = OCRJsonDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/ocr_art/defaults.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        json_path = build_info.json_path
        vis_root = build_info.vis_root

        datasets = dict()

        if not os.path.exists(json_path):
            warnings.warn("json path {} does not exist.".format(json_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            json_path=json_path,
            vis_root=vis_root
        )

        return datasets
