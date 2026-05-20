"""
QLoRA 微调脚本

使用 PEFT 库对基座模型进行 QLoRA 微调。
用法: python src/generation/finetune.py

注意：此脚本需要 GPU 支持（至少 6GB 显存）。
"""

import json
import logging
import os
import sys

import torch
from datasets import Dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from config import (
    BASE_MODEL_NAME,
    TRAINING_DIR,
    FINETUNED_DIR,
    MERGED_MODEL_DIR,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_TARGET_MODULES,
    LEARNING_RATE,
    TRAIN_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    NUM_EPOCHS,
    MAX_SEQ_LENGTH,
    USE_4BIT,
)

logger = logging.getLogger(__name__)


def load_training_data(data_path: str):
    """
    加载微调训练数据。

    数据格式（train.json）：
    [
        {
            "conversations": [
                {"role": "user", "content": "问题"},
                {"role": "assistant", "content": "答案"}
            ],
            "source": "self_constructed"
        }
    ]

    Args:
        data_path: train.json 路径

    Returns:
        Dataset 对象
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"训练数据不存在: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    formatted_texts = []
    for item in raw_data:
        conversations = item.get("conversations", [])
        text_parts = []
        for turn in conversations:
            role = turn.get("role", "")
            content = turn.get("content", "")
            if role == "user":
                text_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                text_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")

        formatted_texts.append("\n".join(text_parts))

    logger.info(f"加载训练数据: {data_path}，共 {len(formatted_texts)} 条")
    return Dataset.from_dict({"text": formatted_texts})


def finetune_with_peft():
    """
    使用 PEFT 库进行 QLoRA 微调：
    1. 加载 train.json 数据集
    2. 用 BitsAndBytesConfig 加载 4bit 量化模型
    3. 配置 LoRA
    4. 训练
    5. 保存 adapter 到 models/finetuned_lora/
    6. 可选：合并并保存到 models/merged_model/
    """
    # 检查 GPU
    if not torch.cuda.is_available():
        logger.error("微调需要 GPU 支持，未检测到 CUDA 设备。")
        print("错误: 微调需要 GPU 支持，未检测到 CUDA 设备。")
        return False

    device = "cuda"
    logger.info(f"使用设备: {device}")

    # 1. 加载数据
    train_path = os.path.join(TRAINING_DIR, "train.json")
    try:
        dataset = load_training_data(train_path)
        if len(dataset) == 0:
            print("错误: 训练数据为空")
            return False
    except FileNotFoundError:
        print(f"错误: 训练数据不存在 ({train_path})")
        print("请先准备 data/training/train.json 格式的训练数据")
        return False

    # 2. 加载模型和分词器
    logger.info(f"正在加载基座模型: {BASE_MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4bit 量化配置
    if USE_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        logger.info("使用 4bit 量化加载模型")
    else:
        bnb_config = None

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
    )

    # 数据预处理
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # 3. 配置 LoRA
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. 训练配置
    training_args = TrainingArguments(
        output_dir=os.path.join(FINETUNED_DIR, "checkpoints"),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="none",
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # 5. 开始训练
    logger.info("开始微调训练...")
    print("\n" + "=" * 50)
    print("  QLoRA 微调训练开始")
    print(f"  训练数据: {len(dataset)} 条")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch Size: {TRAIN_BATCH_SIZE}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print("=" * 50 + "\n")

    trainer.train()

    # 6. 保存 LoRA adapter
    os.makedirs(FINETUNED_DIR, exist_ok=True)
    model.save_pretrained(FINETUNED_DIR)
    tokenizer.save_pretrained(FINETUNED_DIR)
    logger.info(f"LoRA adapter 已保存: {FINETUNED_DIR}")

    # 7. 合并并保存完整模型
    try:
        logger.info("正在合并 LoRA 权重...")
        merged_model = model.merge_and_unload()
        os.makedirs(MERGED_MODEL_DIR, exist_ok=True)
        merged_model.save_pretrained(MERGED_MODEL_DIR)
        tokenizer.save_pretrained(MERGED_MODEL_DIR)
        logger.info(f"合并模型已保存: {MERGED_MODEL_DIR}")
    except Exception as e:
        logger.warning(f"合并模型失败（不影响 adapter 使用）: {e}")

    print("\n" + "=" * 50)
    print("  微调完成！")
    print(f"  LoRA adapter: {FINETUNED_DIR}")
    print(f"  合并模型: {MERGED_MODEL_DIR}")
    print("  使用微调模型: python run.py --model finetuned")
    print("=" * 50)

    return True


if __name__ == "__main__":
    finetune_with_peft()
