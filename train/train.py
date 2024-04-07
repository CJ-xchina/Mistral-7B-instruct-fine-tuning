import os
import shutil

import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
from iscasmodel.core import ModelUtils

model_utils = ModelUtils()


def get_completion(query: str, model, tokenizer) -> str:
    device = "cpu"

    prompt_template = """
  Below is an instruction that describes a task. Write a response that appropriately completes the request.
  ### Question:
  {query}

  ### Answer:
  """
    prompt = prompt_template.format(query=query)

    encodeds = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)

    model_inputs = encodeds.to(device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True,
                                   pad_token_id=tokenizer.eos_token_id)
    decoded = tokenizer.batch_decode(generated_ids)
    return (decoded[0])


def generate_prompt(data_point):
    """
    Generate input text based on a prompt, task instruction, (context info.), and answer
    :param data_point: dict: Data point
    :return: dict: tokenized prompt
    """

    if data_point['input']:
        text = 'Below is an instruction that describes a task, paired with an input that provides' \
               ' further context. Write a response that appropriately completes the request.\n\n'
        text += f'### Instruction:\n{data_point["instruction"]}\n\n'
        text += f'### Input:\n{data_point["input"]}\n\n'
        text += f'### Response:\n{data_point["output"]}'

    else:
        text = 'Below is an instruction that describes a task. Write a response that ' \
               'appropriately completes the request.\n\n'
        text += f'### Instruction:\n{data_point["instruction"]}\n\n'
        text += f'### Response:\n{data_point["output"]}'
    return text


if torch.cuda.is_available():
    print("CUDA is available", flush=True)
    num_devices = torch.cuda.device_count()
    print("Number of CUDA devices: ", num_devices, flush=True)

    for i in range(num_devices):
        print("Device ", i, ": ", torch.cuda.get_device_name(i), flush=True)
else:
    print("CUDA is not available", flush=True)
    num_devices = 1

QUANTIZATION = "4bit"  # DEFINE QUANTIZATION HERE. Choose from ("none" | "8bit" | "4bit")
model_path = model_utils.get_model_file_path()  # 模型路径
dataset_path = model_utils.get_dataset_path()  # 训练数据集路径
tokenizer_path = "../resources/tokenizer"  # tokenizer路径
output_dir = "../output_model"  # 模型输出路径
model_id = "mistralai/Mistral-7B-Instruct-v0.2"  # model's huggingface id
per_device_train_batch_size = 1
per_device_eval_batch_size = 1

# 目标文件夹路径
train_folder = "./tmp_train"

validation_folder = "./tmp_validation"

if not os.path.exists(train_folder):
    os.makedirs(train_folder)

if not os.path.exists(validation_folder):
    os.makedirs(validation_folder)

validation_file = ""

# Pre-define quantization configs
bb_config_4b = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
bb_config_8b = BitsAndBytesConfig(
    load_in_8bit=True,
)


def quantization_config(quantization):
    if quantization == "8bit":
        return bb_config_8b
    else:
        return bb_config_4b


# 下载并加载预训练模型，同时指定quantization_config和device_map，并且设置cache_dir为自定义路径
# model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto", cache_dir=custom_cache_dir)

if QUANTIZATION == "none":
    model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=model_path).to("cuda")
else:
    model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=model_path,
                                                 quantization_config=quantization_config(QUANTIZATION))

# 下载并加载预训练分词器，设置cache_dir为自定义路径，并添加EOS令牌
tokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=True, cache_dir=tokenizer_path,
                                          padding_side="left")

# result = get_completion(query="Will capital gains affect my tax bracket?", model=model, tokenizer=tokenizer)
# print(result)

# 遍历源文件夹中的所有文件
for index, filename in enumerate(os.listdir(dataset_path)):
    # 构建源文件路径
    source_file_path = os.path.join(dataset_path, filename)

    # 检查是否为文件
    if os.path.isfile(source_file_path):
        # 构建目标文件路径并修改文件名为.json
        print(index)
        print(len(os.listdir(dataset_path)))
        if index == len(os.listdir(dataset_path)) - 1:
            destination_file_path = os.path.join(validation_folder, f"{os.path.splitext(filename)[0]}.json")
            print(destination_file_path)
            print(validation_file)
            validation_file = destination_file_path

        else:
            destination_file_path = os.path.join(train_folder, f"{os.path.splitext(filename)[0]}.json")

        # 复制文件并重命名
        shutil.copy(source_file_path, destination_file_path)

        # 打印结果
        print(
            f"文件 '{filename}' 已复制到 '{destination_file_path}' 且文件名已修改为 {os.path.splitext(filename)[0]}.json")

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)


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
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

peft_model = get_peft_model(model, lora_config)
print_trainable_parameters(peft_model)
model.add_adapter(lora_config, adapter_name="adapter")

# Parallelization is possible if system is multi-GPU
if torch.cuda.device_count() > 1:
    model.is_parallelizable = True
    model.model_parallel = True

tokenizer.pad_token = tokenizer.eos_token

ips = model_utils.get_ips()
print(ips)
model_utils.get_name()
master_addr = ips[0]

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_folder,
    eval_dataset=validation_folder,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        warmup_steps=5,
        max_steps=1000,
        learning_rate=2e-4,
        bf16=True if (QUANTIZATION != "8bit") else False,
        fp16=True if (QUANTIZATION == "8bit") else False,
        logging_dir="./logs",
        logging_steps=50,
        output_dir=output_dir,
        optim="paged_adamw_8bit",
        save_strategy="epoch",
        evaluation_strategy="steps",
        eval_steps=50,
        push_to_hub=True,
        do_eval=True
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False
trainer.train()
