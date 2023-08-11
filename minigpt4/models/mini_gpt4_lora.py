import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from minigpt4.common.registry import registry
from minigpt4.models.blip2 import Blip2Base, disabled_train
from minigpt4.models.modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer
import os
from peft import LoraConfig, get_peft_config, get_peft_model

# os.environ['XDG_CACHE_HOME']="/hetu_group/wangyingjun/LargeModel/LAVIS/transformers/.cache"
@registry.register_model("mini_gpt4_lora")
class MiniGPT4LoRA(Blip2Base):
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
        num_query_token=32,
        llama_model="",
        prompt_path="",
        prompt_template="",
        max_txt_len=32,
        end_sym='\n',
        low_resource=False,  # use 8 bit and put vit in cpu
        device_8bit=0, # the device of 8bit model should be set when loading and cannot be changed anymore.
        max_input_txt_len=500,
        use_nlp_model=False,
        use_lora=True
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource
        self.vit_model=vit_model

        print('Loading VIT')
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            if vit_model=="eva_clip_g":
                for name, param in self.ln_vision.named_parameters():
                    param.requires_grad = False
                self.ln_vision = self.ln_vision.eval()
                self.ln_vision.train = disabled_train
            logging.info("freeze vision encoder")
        print('Loading VIT Done')

        print('Loading Q-Former')
        if vit_model!="eva_clip_g":
            self.vit_proj = nn.Linear(
                self.visual_encoder.num_features, 1408
            )

            self.Qformer, self.query_tokens = self.init_Qformer(
                num_query_token, 1408
            )
            # import pdb;pdb.set_trace()
        else:

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
        
        print('Loading LLAMA')
        print("llama_model is {}".format(llama_model))
        self.use_nlp_model=use_nlp_model
        if self.use_nlp_model:
            print("use nlp model")
            DEFAULT_PAD_TOKEN = "[PAD]"
            DEFAULT_EOS_TOKEN = "</s>"
            DEFAULT_BOS_TOKEN = "</s>"
            DEFAULT_UNK_TOKEN = "</s>"
            def update_tokenizer(tokenizer):
                if tokenizer.pad_token is None:
                    special_tokens_dict = {"pad_token": DEFAULT_PAD_TOKEN}
                    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
                tokenizer.add_special_tokens(
                    {
                        "eos_token": DEFAULT_EOS_TOKEN,
                        "bos_token": DEFAULT_BOS_TOKEN,
                        "unk_token": DEFAULT_UNK_TOKEN,
                    }
                )
                return tokenizer

            self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
            self.llama_tokenizer = update_tokenizer(self.llama_tokenizer)
            self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token


            print("low_resource is {}".format(self.low_resource))
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
        else:
            print("not use nlp model")
            self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
            self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
            print("low_resource is {}".format(self.low_resource))
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
        
        self.use_lora = use_lora
        if self.use_lora:

            print("Loading LoRA")
            inference_mode = False
            lora_r = 8
            lora_alpha = 32
            lora_dropout = 0.05
            peft_config = LoraConfig(
                target_modules=r'.*model.*\.(q_proj|v_proj)',
                inference_mode=inference_mode,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout
            )
            self.llama_model = get_peft_model(self.llama_model, peft_config)
            self.llama_model.print_trainable_parameters()
            print("Loading LoRA Done")

#         print('Loading LLAMA')
#         self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
#         self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

#         if self.low_resource:
#             self.llama_model = LlamaForCausalLM.from_pretrained(
#                 llama_model,
#                 torch_dtype=torch.float16,
#                 load_in_8bit=True,
#                 device_map={'': device_8bit}
#             )
#         else:
#             self.llama_model = LlamaForCausalLM.from_pretrained(
#                 llama_model,
#                 torch_dtype=torch.float16,
#             )

#         for name, param in self.llama_model.named_parameters():
#             param.requires_grad = False
#         print('Loading LLAMA Done')

        self.llama_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llama_model.config.hidden_size
        )
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym
        self.max_input_txt_len = max_input_txt_len

        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<ImageHere>" in raw_prompt]
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt Example \n{}'.format(random.choice(self.prompt_list)))
        else:
            self.prompt_list = []

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
            # import pdb;pdb.set_trace()
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device) if self.vit_model=="eva_clip_g" else self.ln_vision(self.vit_proj(self.visual_encoder(image))).to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_llama = self.llama_proj(query_output.last_hidden_state)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)
        return inputs_llama, atts_llama

    def prompt_wrap(self, img_embeds, atts_img, prompt, user_prompt=None):
        # import pdb;pdb.set_trace()
        if prompt:
            batch_size = img_embeds.shape[0]
            p_before, p_after = prompt.split('<ImageHere>')
            p_before_tokens = self.llama_tokenizer(
                p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_after_tokens = self.llama_tokenizer(
                p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            if self.use_lora:
                p_before_embeds = self.llama_model.model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
                p_after_embeds = self.llama_model.model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
            else:
                p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
                p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)

            if user_prompt is not None:
                # print(user_prompt)
                user_prompt_tokens = self.llama_tokenizer(
                    user_prompt, return_tensors="pt", add_special_tokens=False, padding='max_length', truncation=True, max_length=self.max_input_txt_len).to(img_embeds.device)
                
                if self.use_lora:
                    user_prompt_embeds = self.llama_model.model.model.embed_tokens(user_prompt_tokens.input_ids)
                else:
                    user_prompt_embeds = self.llama_model.model.embed_tokens(user_prompt_tokens.input_ids)

                wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds, user_prompt_embeds], dim=1)
                # wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
                # print('user_prompt_tokens.input_ids.shape',user_prompt_tokens.input_ids.shape)

                wrapped_atts_img = torch.cat([p_before_tokens.attention_mask.expand(batch_size, -1), atts_img, p_after_tokens.attention_mask.expand(batch_size, -1), user_prompt_tokens.attention_mask], dim=1)

                return wrapped_img_embeds, wrapped_atts_img

            wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
            wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
            return wrapped_img_embeds, wrapped_atts_img
        else:
            return img_embeds, atts_img

    def forward(self, samples):
        # import pdb;pdb.set_trace()
        image = samples["image"]
        img_embeds, atts_img = self.encode_img(image)
        if hasattr(samples, 'question_split'):  # VQA dataset
            # print('Here')
            print('VQA Batch')
            vqa_prompt = '###Human: <Img><ImageHere></Img> '
            img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, vqa_prompt)
        elif samples.__contains__('user_prompt'):
            prompt = '###Human: <Img><ImageHere></Img> '
            user_prompt = samples['user_prompt']
            # if random.random() < 0.02:
            #     print(len(user_prompt[0]))
            # print(len(user_prompt))
            # print(user_prompt,samples["text_input"])
            img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, prompt, user_prompt)

        elif self.prompt_list:
            prompt = random.choice(self.prompt_list)
            # print('Here, the selected prompt is ' + prompt)
            img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, prompt)

        self.llama_tokenizer.padding_side = "right"

        text = [t + self.end_sym for t in samples["text_input"]]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(image.device)

        # print('to_regress_tokens.input_ids.shape',to_regress_tokens.input_ids.shape)

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )

        empty_targets = (
            torch.ones([atts_img.shape[0], atts_img.shape[1]+1],
                       dtype=torch.long).to(image.device).fill_(-100)  # plus one for bos
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        if self.use_lora:
            bos_embeds = self.llama_model.model.model.embed_tokens(bos)
        else:
            bos_embeds = self.llama_model.model.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        if self.use_lora:
            to_regress_embeds = self.llama_model.model.model.embed_tokens(to_regress_tokens.input_ids)
        else:
            to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)

        test_input = {
            'inputs_embeds':inputs_embeds,
            'attention_mask':attention_mask,
            'return_dict':True,
            'labels':targets,
        }

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss

        return {"loss": loss}

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        q_former_model = cfg.get("q_former_model", "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llama_model = cfg.get("llama_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        freeze_qformer = cfg.get("freeze_qformer", True)
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)

        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')
        max_input_txt_len = cfg.get("max_input_txt_len", 500)
        use_nlp_model = cfg.get("use_nlp_model", False)
        use_lora = cfg.get("use_lora", True)

        model = cls(
            vit_model=vit_model,
            q_former_model=q_former_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            freeze_qformer=freeze_qformer,
            num_query_token=num_query_token,
            llama_model=llama_model,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
            max_input_txt_len=max_input_txt_len,
            use_nlp_model=use_nlp_model,
            use_lora=use_lora
        )
        
        ckpt_path = cfg.get("ckpt_before", "")  # load weights of MiniGPT-4
        if ckpt_path:
            print("Load Previous BLIP2-LLM Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)

        # ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        # if ckpt_path:
        #     print("Load BLIP2-LLM Checkpoint: {}".format(ckpt_path))
        #     ckpt = torch.load(ckpt_path, map_location="cpu")
        #     msg = model.load_state_dict(ckpt['model'], strict=False)
        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            print("Load BLIP2-LLM Checkpoint: {}".format(ckpt_path))

            if use_lora:
                from collections import OrderedDict
                state_dict = OrderedDict()
                model_dict = model.state_dict()

                for key in model_dict.keys():
                    print("model param: ", key)
                for key in ckpt['model'].keys():
                    print("ckpt param: ", key)
                
                model_lora = [k for k in model_dict if 'lora' in k]
                ckpt_lora = [k for k in ckpt['model'] if 'lora' in k]
                print("model lora param: ", model_lora)
                print("ckpt lora param: ", ckpt_lora)
                
                for key, value in ckpt['model'].items():
                    # model.llama_model.base_model.model.model.layers
                    # llama_model.model.layers.25.self_attn.rotary_emb.inv_freq
                    before = "llama_model.model.layers"
                    after = "llama_model.base_model.model.model.layers"
                    new_key = key.replace(before, after)
                    
                    before1, before2 = "lora_A.weight", "lora_B.weight"
                    after1, after2 = "lora_A.default.weight", "lora_B.default.weight"
                    new_key = new_key.replace(before1, after1).replace(before2, after2)
                    #print("before key: ", key)
                    #print("after key: ", new_key)
                    # new_key = key
                    if new_key in model_dict and value.shape == model_dict[new_key].shape:
                        print("match: {}".format(new_key))
                        state_dict[new_key] = value
                    elif new_key not in model_dict:
                        print("!!!!!!!!!!!!!!! cannot find key: {}".format(new_key))
                    else:
                        print("!!!!!!!!!!!!!!! can find key: {} but shape not match".format(new_key))
                msg = model.load_state_dict(state_dict, strict=False)
            else:
                for key in ckpt['model'].keys():
                    print("ckpt param: ", key)
                msg = model.load_state_dict(ckpt['model'], strict=False)

        return model
