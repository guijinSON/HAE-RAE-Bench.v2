from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
import torch

def get_llama(
    hf_ckpt="beomi/llama-2-ko-7b",
    torch_dtype=torch.bfloat16,
    lora_r = 64,
    lora_alpha = 32,
    lora_target_modules = ["embed_tokens","q_proj","k_proj"],
    lora_dropout = 0.05,
    lora_bias = "none",
    lora_task_type = "CAUSAL_LM"
    ):
  
    tokenizer = AutoTokenizer.from_pretrained(hf_ckpt)
    model = AutoModelForCausalLM.from_pretrained(hf_ckpt,torch_dtype = torch_dtype)

    lora_config = LoraConfig(
                            r=lora_r,
                            lora_alpha=lora_alpha,
                            target_modules=lora_target_modules,
                            lora_dropout=lora_dropout,
                            bias=lora_bias,
                            task_type=lora_task_type,
                            )
    
    model = get_peft_model(model, lora_config)

    return model,tokenizer
