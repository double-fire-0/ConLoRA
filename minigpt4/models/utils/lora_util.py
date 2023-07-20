from peft import LoraConfig, get_peft_model


# All these default config should not be the best pratice

default_lora_config_ve = LoraConfig(
    r=16,  # larger may be good for ve, we choose from [16, 32]
    lora_alpha=16,
    target_modules=r'.*blocks\.\b(1[4-9]|2[0-9]|3[0-8])\b\..*(\.query|\.value)',  # blocks from 14 to 38(last block)
    lora_dropout=0.1,  # effect is not clear
    bias="none",  # may be we should add more param for ve
)

default_lora_config_qformer = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=['.query','.value'],
    lora_dropout=0.1,
    bias="none",
)

# just a copy
default_lora_config_llm = LoraConfig(
    target_modules=r'.*model.*\.(q_proj|v_proj)',
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
)

default_lora_config_map = {
    've': default_lora_config_ve,
    'qformer': default_lora_config_qformer,
    'llm': default_lora_config_llm,
}


def get_lora_model(model, config):
    if not hasattr(model, 'config'):
        model.config = None
    return get_peft_model(model, config)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )