from minigpt4.runners.runner_base import RunnerBase
from minigpt4.common.dist_utils import main_process
from minigpt4.common.registry import registry
import os
import logging
import torch

@registry.register_runner("runner_lora")
class RunnerRola(RunnerBase):

    @main_process
    def _save_checkpoint(self, cur_epoch, is_best=False):
        """
        Save the checkpoint at the current epoch.
        """
        save_to_lora = os.path.join(
            self.output_dir,
            "lora_{}".format("best" if is_best else cur_epoch),
        )
        model_no_ddp = self.unwrap_dist_model(self.model)
        model_no_ddp.save_lora(save_to_lora)

        # checkpoint may not be used but we save it for safety
        param_grad_dic = {
            k: v.requires_grad for (k, v) in model_no_ddp.named_parameters()
        }
        state_dict = model_no_ddp.state_dict()
        for k in list(state_dict.keys()):
            if k in param_grad_dic.keys() and not param_grad_dic[k]:
                # delete parameters that do not require gradient
                del state_dict[k]
        save_obj = {
            "model": state_dict,
            "optimizer": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "epoch": cur_epoch,
        }
        save_to = os.path.join(
            self.output_dir,
            "checkpoint_{}.pth".format("best" if is_best else cur_epoch),
        )
        logging.info("Saving checkpoint at epoch {} to {}.".format(cur_epoch, save_to))
        torch.save(save_obj, save_to)
    
