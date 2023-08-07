# TODO: Make it work with all kind of data

import logging
import random
import copy
import os

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
from peft import PeftModel

from minigpt4.common.registry import registry
from minigpt4.models.blip2 import Blip2Base, disabled_train
from minigpt4.models.modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList
from minigpt4.conversation.conversation import StoppingCriteriaSub
from minigpt4.models.utils.lora_util import default_lora_config_map, get_lora_model, print_trainable_parameters


@registry.register_model("conlora_uin")
class ConLora_UIN(Blip2Base):
    """
    BLIP2 GPT-LLAMA model.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/minigpt4.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        freeze_qformer=True,
        freeze_fc=True,
        num_query_token=32,
        llama_model="",
        prompt_path="",
        prompt_template="",   # '###Human: {} ###Assistant: '
        max_txt_len=32,
        end_sym='\n',
        low_resource=False,  # use 8 bit and put vit in cpu
        # the device of 8bit model should be set when loading and cannot be changed anymore.
        device_8bit=0,
        content_use_lora=['ve', 'qformer'],
        lora_config_map=default_lora_config_map,
    ):
        super().__init__()

        assert set(content_use_lora).issubset(set(['ve', 'qformer', 'llm']))
        self.content_use_lora = content_use_lora
        self.lora_config_map = lora_config_map

        self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource

        print('Loading VIT')
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision = self.ln_vision.eval()
            self.ln_vision.train = disabled_train
            logging.info("freeze vision encoder")
        print('Loading VIT Done')

        print('Loading Q-Former')
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.load_from_pretrained(url_or_filename=q_former_model)

        if freeze_qformer:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
            self.query_tokens.requires_grad = False
            logging.info("freeze Qformer")
        print('Loading Q-Former Done')

        # postiton tag for load llama
        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(
            llama_model, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        # debug = True
        # if debug:
        #     self.llama_model = torch.nn.modules.Identity()

        if self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={'': device_8bit}
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
            )

        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        print('Loading LLAMA Done')


        self.llama_proj = nn.Linear(
            self.Qformer.config.hidden_size, 4096  # self.llama_model.config.hidden_size
        )
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

        # freeze fc before set lora
        if freeze_fc:
            for name, param in self.llama_proj.named_parameters():
                param.requires_grad = False
            self.llama_proj = self.llama_proj.eval()
            self.llama_proj.train = disabled_train
            logging.info("freeze Qformer")

        # Set up Lora model
        self.lora_content_map = {'ve': self.visual_encoder, 'qformer': self.Qformer, 'llm': self.llama_model}
        self._init_lora()
        self.current_lora = 'default'


        # from ipdb import set_trace
        # set_trace()
        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [
                raw_prompt for raw_prompt in raw_prompts if "<ImageHere>" in raw_prompt]
            self.prompt_list = [prompt_template.format(
                p) for p in filted_prompts]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt Example \n{}'.format(
                random.choice(self.prompt_list)))
        else:
            self.prompt_list = []

    def _init_lora(self):
        for content in self.content_use_lora:
            # from ipdb import set_trace; set_trace()
            if content in self.lora_content_map.keys():
                self.lora_content_map[content] = get_lora_model(self.lora_content_map[content], self.lora_config_map[content])
            if content == 've':
                self.visual_encoder = self.lora_content_map[content]
            elif content == 'qformer':
                self.Qformer = self.lora_content_map[content]
            elif content == 'llm':
                self.llama_model = self.lora_content_map[content]
            else:
                raise NotImplementedError(
                    f'content {content} not supported in conlora')

            # if content == 've':
            #     self.visual_encoder = get_lora_model(
            #         self.visual_encoder, self.lora_config_map[content])
            #     print('Set up Lora for visual encoder')
            #     print_trainable_parameters(self.visual_encoder)
            # elif content == 'qformer':
            #     self.Qformer = get_lora_model(
            #         self.Qformer, self.lora_config_map[content])
            #     print('Set up Lora for Qformer')
            #     print_trainable_parameters(self.Qformer)
            # elif content == 'llm':
            #     self.llama_model = get_lora_model(
            #         self.llama_model, self.lora_config_map[content])
            #     print('Set up Lora for LLAMA')
            #     print_trainable_parameters(self.llama_model)
            # else:
            #     raise NotImplementedError(
            #         f'content {content} not supported in conlora')


    def vit_to_cpu(self):
        self.ln_vision.to("cpu")
        self.ln_vision.float()
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()

    def encode_img(self, image):
        device = image.device
        if self.low_resource:
            self.vit_to_cpu()
            image = image.to("cpu")

        with self.maybe_autocast():
            image_embeds = self.ln_vision(
                self.visual_encoder(image)).to(device)
            image_atts = torch.ones(
                image_embeds.size()[:-1], dtype=torch.long).to(device)

            query_tokens = self.query_tokens.expand(
                image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_llama = self.llama_proj(query_output.last_hidden_state)
            atts_llama = torch.ones(
                inputs_llama.size()[:-1], dtype=torch.long).to(image.device)
        return inputs_llama, atts_llama

    def prompt_wrap(self, img_embeds, atts_img, input_item, output_item, prompt):
        if prompt:
            # from ipdb import set_trace
            # set_trace()

            batch_size = img_embeds.shape[0]
            prompts = [prompt.format(d) for d in input_item]
            examples = [prompts[i] + output_item[i] + self.end_sym for i in range(len(prompts))]

            p_before, p_after = [p.split('ImageHere')[0] for p in prompts], [p.split('ImageHere')[-1] for p in prompts]
            e_before, e_after = [e.split('ImageHere')[0] for e in examples], [e.split('ImageHere')[-1] for e in examples]

            p_before_tokens = self.llama_tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            # p_after_tokens = self.llama_tokenizer(
            #     p_after, return_tensors="pt", add_special_tokens=False, padding="longest", truncation=True, max_length=self.max_txt_len).to(img_embeds.device)

            p_after_tokens_nopad = self.llama_tokenizer(p_after, add_special_tokens=False)
            e_after_tokens = self.llama_tokenizer(
                e_after, return_tensors="pt", add_special_tokens=False, padding="longest", truncation=True, max_length=self.max_txt_len).to(img_embeds.device)
            labels_after_image = copy.deepcopy(e_after_tokens)

            # TODO find more reliable way to handle it
            for index, label in enumerate(labels_after_image.input_ids):
                label[: len(p_after_tokens_nopad.input_ids[index]) - 1] = -100  # -1 for bug in ": <" spae bug in tokenizer
            labels_after_image = labels_after_image.input_ids.masked_fill(
                labels_after_image.input_ids == self.llama_tokenizer.pad_token_id, -100
            )

            p_before_embeds = self.get_embed_tokens(p_before_tokens.input_ids)  # .expand(batch_size, -1, -1)
            # .expand(batch_size, -1, -1) expand shouldn't make influence
            e_after_embeds = self.get_embed_tokens(e_after_tokens.input_ids)
            wrapped_all_embeds = torch.cat([p_before_embeds, img_embeds, e_after_embeds], dim=1)
            wrapped_atts_all = torch.cat([atts_img[:, :1].expand(
                -1, p_before_embeds.shape[1] + img_embeds.shape[1]), e_after_tokens.attention_mask], dim=1)

            empty_labels = torch.ones([img_embeds.shape[0], p_before_embeds.shape[1] + img_embeds.shape[1] + 1],
                                      dtype=torch.long).to(img_embeds.device).fill_(-100)  # plus one for bos
            labels = torch.cat([empty_labels, labels_after_image], dim=1)

            # return all part expect bos
            return wrapped_all_embeds, wrapped_atts_all, labels
        else:
            assert NotImplementedError
    
    def prompt_wrap_generate(self, img_embeds, atts_img, input_item, prompt):
        if prompt:
            # from ipdb import set_trace
            # set_trace()

            one_image_embed = img_embeds[0].unsqueeze(0)
            prompts = [prompt.format(d) for d in input_item]

            p_before, p_after = [p.split('ImageHere')[0] for p in prompts], [p.split('ImageHere')[-1] for p in prompts]
            p_before, p_after = p_before[0], p_after[0]

            p_before_tokens = self.llama_tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_after_tokens = self.llama_tokenizer(p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)

            p_before_embeds = self.get_embed_tokens(p_before_tokens.input_ids)
            p_after_embeds = self.get_embed_tokens(p_after_tokens.input_ids)

            wrapped_generate_embeds = torch.cat([p_before_embeds, one_image_embed, p_after_embeds], dim=1)

            # return all part expect bos
            return wrapped_generate_embeds
        else:
            assert NotImplementedError

    def forward(self, samples):
        image = samples["image"]
        img_embeds, atts_img = self.encode_img(image)

        input_item = samples["input_item"]
        output_item = samples["output_item"]

        self.llama_tokenizer.padding_side = "right"

        if hasattr(samples, 'question_split'):  # VQA dataset
            print('VQA Batch')
            vqa_prompt = '###Human: <Img><ImageHere></Img> '
            wrapped_embeds_all, wrapped_atts_all, labels = self.prompt_wrap(img_embeds, atts_img, input_item, output_item, vqa_prompt)
        elif self.prompt_list:
            prompt = random.choice(self.prompt_list)
            wrapped_embeds_all, wrapped_atts_all, labels = self.prompt_wrap(img_embeds, atts_img, input_item, output_item, prompt)

        batch_size = img_embeds.shape[0]

        bos = torch.ones([batch_size, 1], dtype=labels.dtype, device=labels.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.get_embed_tokens(bos)
        atts_bos = wrapped_atts_all[:, :1]

        inputs_embeds = torch.cat([bos_embeds, wrapped_embeds_all], dim=1)
        attention_mask = torch.cat([atts_bos, wrapped_atts_all], dim=1)

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=labels,
            )
            
        loss = outputs.loss

        return {"loss": loss}

    def generate(self, samples):
        image = samples["image"]
        img_embeds, atts_img = self.encode_img(image)

        input_item = samples["input_item"]
        output_item = samples["output_item"]
        self.llama_tokenizer.padding_side = "right"

        if hasattr(samples, 'question_split'):  # VQA dataset
            print('VQA Batch')
            vqa_prompt = '###Human: <Img><ImageHere></Img> '
            inputs_embeds = self.prompt_wrap_generate(img_embeds, atts_img, input_item, vqa_prompt)
        elif self.prompt_list:
            prompt = random.choice(self.prompt_list)
            inputs_embeds = self.prompt_wrap_generate(img_embeds, atts_img, input_item, prompt)

        batch_size = img_embeds.shape[0]

        bos = torch.ones([1, 1], dtype=torch.long, device=inputs_embeds.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.get_embed_tokens(bos)
        inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=1)

        stop_words_ids = [torch.tensor([835]).to(self.device),
                    torch.tensor([2277, 29937]).to(self.device)]  # '###' can be encoded in two different ways.
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

        with self.maybe_autocast():
            outputs = self.llama_model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=300,
                stopping_criteria=stopping_criteria,
                num_beams=1,
                do_sample=True,
                min_length=1,
                top_p=0.9,
                repetition_penalty=1.0,
                length_penalty=0,  # length penalty for detection is unknown
                temperature=1,
            )
        output_token = outputs[0]
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
        output_text = self.llama_tokenizer.decode(output_token, add_special_tokens=False)
        
        return {'input_item': input_item[0], "output_text": output_text, 'label': output_item[0], 'image_tensor': image[0]}
    
    def get_embed_tokens(self, samples):
        if isinstance(self.llama_model, PeftModel):
            return self.llama_model.model.model.embed_tokens(samples)
        else:
            return self.llama_model.model.embed_tokens(samples)

    def save_lora(self, path):
        for content in self.content_use_lora:
            save_path = os.path.join(path, content)
            self.lora_content_map[content].save_pretrained(save_path)

            # if content == 've':
            #     self.visual_encoder.save_pretrained(save_path)
            # elif content == 'qformer':
            #     self.Qformer.save_pretrained(save_path)
            # elif content == 'llm':
            #     self.llama_model.save_pretrained(save_path)
            # else:
            #     raise NotImplementedError

    def load_lora(self, name, path):
        for content in self.content_use_lora:
            load_path = os.path.join(path, content)
            self.lora_content_map[content].load_adapter(load_path, adapter_name=name)

            # if content == 've':
            #     self.visual_encoder.load_adapter(load_path, adapter_name=name)
            # elif content == 'qformer':
            #     self.Qformer.load_adapter(load_path, adapter_name=name)
            # elif content == 'llm':
            #     self.llama_model.load_adapter(load_path, adapter_name=name)
            # else:
            #     raise NotImplementedError

    # TODO add lora now class attribute
    # TODO: Check load adpater and reload model

    def check_set_lora(self, lora_name):
        if lora_name == self.current_lora:
            print('Keep the same with {lora_name}')
            pass
        elif lora_name == 'original':
            for content in self.content_use_lora:
                self.lora_content_map[content].base_model.disable_adapter_layers()
                # if content == 've':
                #     self.visual_encoder.base_model.disable_adapter_layers()
                # elif content == 'qformer':
                #     self.Qformer.base_model.disable_adapter_layers()
                # elif content == 'llm':
                #     self.llama_model.base_model.disable_adapter_layers()
                # else:
                #     raise NotImplementedError
            print('Set LORA to {}'.format(lora_name))
        else:
            for content in self.content_use_lora:
                self.lora_content_map[content].base_model.enable_adapter_layers()
                self.lora_content_map[content].set_adapter(lora_name)

                # if content == 've':
                #     self.visual_encoder.base_model.enable_adapter_layers()
                #     self.visual_encoder.set_adapter(lora_name)
                # elif content == 'qformer':
                #     self.Qformer.base_model.enable_adapter_layers()
                #     self.Qformer.set_adapter(lora_name)
                # elif content == 'llm':
                #     self.llama_model.base_model.enable_adapter_layers()
                #     self.llama_model.set_adapter(lora_name)
                # else:
                #     raise NotImplementedError
            print('Set LORA to {}'.format(lora_name))

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        q_former_model = cfg.get(
            "q_former_model", "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llama_model = cfg.get("llama_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        freeze_qformer = cfg.get("freeze_qformer", True)
        freeze_fc = cfg.get("freeze_fc", True)
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)

        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')

        # from ipdb import set_trace; set_trace()
        content_use_lora = cfg.get("content_use_lora", ['ve', 'qformer'])  # not use lora llm by default

        # TODO: copy bug of loraconfig fix next peft version
        # lora_config_map = copy.deepcopy(default_lora_config_map)
        lora_config_map = default_lora_config_map
        for content in content_use_lora:
            content_cfg = cfg.get(content + '_lora_cfg', None)
            if content_cfg is not None:
                for key, item in content_cfg.items():
                    setattr(lora_config_map[content], key, item)
                print(f'Reset lora config of {content} to \n{lora_config_map[content]}')

        model = cls(
            vit_model=vit_model,
            q_former_model=q_former_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            freeze_qformer=freeze_qformer,
            freeze_fc=freeze_fc,
            num_query_token=num_query_token,
            llama_model=llama_model,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
            content_use_lora=content_use_lora,
            lora_config_map=lora_config_map
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            print("Load BLIP2-LLM Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)
        
        lora_path = cfg.get('lora_path',{})
        for name, path in lora_path.items():
            model.load_lora(name, path)

        return model

