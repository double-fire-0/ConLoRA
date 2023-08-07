"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import os

import torch
import torch.distributed as dist
from minigpt4.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
from minigpt4.common.logger import MetricLogger, SmoothedValue
from minigpt4.common.registry import registry
from minigpt4.datasets.data_utils import prepare_sample
import torchvision.transforms as T
import base64
from io import BytesIO
import csv
import copy
from torchvision.utils import draw_bounding_boxes
import re
import numpy as np
import cv2

class BaseTask:
    def __init__(self, **kwargs):
        super().__init__()

        self.inst_id_key = "instance_id"
        self.tmp_res_path = '/hetu_group/wangyan/tunnel/res_degpt/'
        self.tensorboard_writer = None

    @classmethod
    def setup_task(cls, **kwargs):
        return cls()

    def set_tensorboard(self, tensorboard_writer):
        self.tensorboard_writer = tensorboard_writer

    def build_model(self, cfg):
        model_config = cfg.model_cfg

        model_cls = registry.get_model_class(model_config.arch)
        return model_cls.from_config(model_config)

    def build_datasets(self, cfg):
        """
        Build a dictionary of datasets, keyed by split 'train', 'valid', 'test'.
        Download dataset and annotations automatically if not exist.

        Args:
            cfg (common.config.Config): _description_

        Returns:
            dict: Dictionary of torch.utils.data.Dataset objects by split.
        """

        datasets = dict()

        datasets_config = cfg.datasets_cfg

        assert len(datasets_config) > 0, "At least one dataset has to be specified."

        for name in datasets_config:
            dataset_config = datasets_config[name]

            builder = registry.get_builder_class(name)(dataset_config)
            dataset = builder.build_datasets()

            dataset['train'].name = name
            if 'sample_ratio' in dataset_config:
                dataset['train'].sample_ratio = dataset_config.sample_ratio

            datasets[name] = dataset

        return datasets

    def train_step(self, model, samples):
        loss = model(samples)["loss"]
        return loss

    def valid_step(self, model, samples):
        raise NotImplementedError

    def before_evaluation(self, model, dataset, **kwargs):
        model.before_evaluation(dataset=dataset, task_type=type(self))

    def after_evaluation(self, **kwargs):
        pass

    def inference_step(self):
        raise NotImplementedError

    def evaluation(self, model, data_loader, cuda_enabled=True):
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        # TODO make it configurable
        print_freq = 10

        results = []

        for samples in metric_logger.log_every(data_loader, print_freq, header):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

            eval_output = self.valid_step(model=model, samples=samples)
            results.extend(eval_output)

        if is_dist_avail_and_initialized():
            dist.barrier()

        return results

    def train_epoch(
        self,
        epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            iters_per_epoch=lr_scheduler.iters_per_epoch,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
        )

    def train_iters(
        self,
        epoch,
        start_iters,
        iters_per_inner_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            start_iters=start_iters,
            iters_per_epoch=iters_per_inner_epoch,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
        )

    def _train_inner_loop(
        self,
        epoch,
        iters_per_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        start_iters=None,
        log_freq=50,
        cuda_enabled=False,
        accum_grad_iters=1,
    ):
        """
        An inner training loop compatible with both epoch-based and iter-based training.

        When using epoch-based, training stops after one epoch; when using iter-based,
        training stops after #iters_per_epoch iterations.
        """
        tb_freq = 500
        use_amp = scaler is not None

        if not hasattr(data_loader, "__next__"):
            # convert to iterator if not already
            data_loader = iter(data_loader)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        # if iter-based runner, schedule lr based on inner epoch.
        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, iters_per_epoch
            )
        )
        header = "Train: data epoch: [{}]".format(epoch)
        if start_iters is None:
            # epoch-based runner
            inner_epoch = epoch
        else:
            # In iter-based runner, we schedule the learning rate based on iterations.
            inner_epoch = start_iters // iters_per_epoch
            header = header + "; inner epoch [{}]".format(inner_epoch)

        for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
            # if using iter-based runner, we stop after iters_per_epoch iterations.
            if i >= iters_per_epoch:
                break

            samples = next(data_loader)

            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            samples.update(
                {
                    "epoch": inner_epoch,
                    "num_iters_per_epoch": iters_per_epoch,
                    "iters": i,
                }
            )
            recoard_sample = copy.deepcopy(samples)

            lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = self.train_step(model=model, samples=samples)

            # after_train_step()
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # update gradients every accum_grad_iters iterations
            if (i + 1) % accum_grad_iters == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()                     
                else:    
                    optimizer.step()
                optimizer.zero_grad()

            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            
            if (i + 1) % log_freq == 0 and self.tensorboard_writer is not None:
                total_iter = inner_epoch * iters_per_epoch + i
                self.tensorboard_writer.add_scalar('train loss', loss, total_iter)
                self.tensorboard_writer.add_scalar('train lr', optimizer.param_groups[0]["lr"], total_iter)

        # TODO fix style
            if (i + 1) % tb_freq == 0 and self.tensorboard_writer is not None:
                # TODO a better style
                # write train images to tensorboard
                # image_norm = samples['image']
                # image_unnorm = self.unnorm_image(image_norm)
                # self.tensorboard_writer.add_images(f'train batch image in rank {get_rank()}', image_unnorm, total_iter)
                try:
                    vis_out = model.module.generate(samples)
                    output_text = vis_out['output_text']
                    # output_text = '<bin_74> <bin_23> <bin_34> <bin_58>###'
                    description = vis_out['description']
                    bin_boxes = vis_out['bin_boxes']
                    image = vis_out['image_tensor']
                    
                    self.tensorboard_writer.add_text(f'text result train in rank {get_rank()}', output_text, total_iter)
                    self.tensorboard_writer.flush()

                    image = self.unnorm_image(image)
                    image_bbox = self.get_image_with_bbox((image * 255).to(torch.uint8), output_text, bin_boxes)
                    image_bbox_des = self.put_text(image_bbox, description)
                    self.tensorboard_writer.add_image(f'image result train in rank {get_rank()}', image_bbox_des, total_iter, dataformats='HWC')
                    
                    samples = next(data_loader)
                    samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

                    vis_out = model.module.generate(samples)
                    output_text = vis_out['output_text']
                    # output_text = '<bin_74> <bin_23> <bin_34> <bin_58>###'
                    description = vis_out['description']
                    bin_boxes = vis_out['bin_boxes']
                    image = vis_out['image_tensor']
                    
                    self.tensorboard_writer.add_text(f'text result eval in rank {get_rank()}', output_text, total_iter)
                    image = self.unnorm_image(image)
                    image_bbox = self.get_image_with_bbox((image * 255).to(torch.uint8), output_text, bin_boxes)
                    image_bbox_des = self.put_text(image_bbox, description)
                    self.tensorboard_writer.add_image(f'image result eval in rank {get_rank()}', image_bbox_des, total_iter, dataformats='HWC')
                    
                    # self.tmp_save_result_wy(vis_out, os.path.join(self.tmp_res_path, 'eval.tsv'))
                except Exception as error:
                    # we don't want it to block training
                    logging.exception("message")

        # after train_epoch()
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }

    def get_image_with_bbox(self, image, output_text, bin_boxes):
        try:
            output_box = re.findall(r"<bin_(.+?)>", output_text)
            output_box = [int(out) for out in output_box]
            bin_boxes = re.findall(r"<bin_(.+?)>", bin_boxes)
            bin_boxes = [int(out) for out in bin_boxes]
            box = [bin_boxes, output_box]
            box = torch.tensor(box, dtype=torch.int)
            image = draw_bounding_boxes(image, box, width=5, colors=["blue", "orange"], fill=False)
            return image  # [Tensor(C,H,W)] uint8 
        except Exception as error:
            # we don't want it to block training
            logging.exception(f"get_image_with_bbox rank: {get_rank()} input: {output_text}")
            raise error

    def put_text(self, img, texts):
        try:
            arrimage = img.numpy().transpose(1, 2, 0)
            result = cv2.putText(arrimage, str(texts), (0, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2)
            return result
        except Exception as error:
            logging.exception("put_text")
            raise error
    
    # TODO a better style
    def unnorm_image(self, image_norm):
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

    # def tmp_save_result_wy(self, out_dict, res_path):
    #     try:
    #         output_text = out_dict['output_text']
    #         description = out_dict['description']
    #         bin_boxes = out_dict['bin_boxes']
    #         image = T.ToPILImage()(out_dict['image_tensor'])
    #         buffered = BytesIO()
    #         image.save(buffered, format="JPEG")
    #         img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    #         with open(res_path, 'a') as out_file:
    #             tsv_writer = csv.writer(out_file, delimiter='\t')
    #             tsv_writer.writerow([img_b64, description, output_text, bin_boxes])
    #     except Exception as error:
    #         print("An exception occurred:", error)

    @staticmethod
    def save_result(result, result_dir, filename, remove_duplicate=""):
        import json

        result_file = os.path.join(
            result_dir, "%s_rank%d.json" % (filename, get_rank())
        )
        final_result_file = os.path.join(result_dir, "%s.json" % filename)

        json.dump(result, open(result_file, "w"))

        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            logging.warning("rank %d starts merging results." % get_rank())
            # combine results from all processes
            result = []

            for rank in range(get_world_size()):
                result_file = os.path.join(
                    result_dir, "%s_rank%d.json" % (filename, rank)
                )
                res = json.load(open(result_file, "r"))
                result += res

            if remove_duplicate:
                result_new = []
                id_list = []
                for res in result:
                    if res[remove_duplicate] not in id_list:
                        id_list.append(res[remove_duplicate])
                        result_new.append(res)
                result = result_new

            json.dump(result, open(final_result_file, "w"))
            print("result file saved to %s" % final_result_file)

        return final_result_file
