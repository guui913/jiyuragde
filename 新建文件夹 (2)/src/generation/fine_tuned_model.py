"""
微调模型加载模块

加载LoRA微调后的模型。如果adapter不存在则自动降级为基座模型。
"""

import logging
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from config import BASE_MODEL_NAME, SYSTEM_PROMPT, FINETUNED_DIR, MERGED_MODEL_DIR, TEMPERATURE, TOP_P, MAX_NEW_TOKENS, REPETITION_PENALTY
from src.generation.prompt_utils import format_chatml_prompt

logger = logging.getLogger(__name__)


class FineTunedModelGenerator:
    """微调模型生成器，加载LoRA adapter"""

    def __init__(self, base_model=None, lora_path: str = FINETUNED_DIR):
        """
        加载微调模型。

        Args:
            base_model: 基座模型生成器实例（可选，不传则新建）
            lora_path: LoRA adapter路径
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.lora_path = lora_path
        self.merged_path = MERGED_MODEL_DIR
        self.model = None
        self.tokenizer = None
        self._is_finetuned = False

        try:
            # 优先尝试加载合并后的完整模型
            if os.path.exists(self.merged_path) and os.listdir(self.merged_path):
                logger.info(f"检测到合并后的微调模型: {self.merged_path}")
                self._load_merged_model()
            # 其次尝试加载LoRA adapter
            elif os.path.exists(self.lora_path) and os.listdir(self.lora_path):
                logger.info(f"检测到LoRA adapter: {self.lora_path}")
                self._load_lora_adapter()
            else:
                logger.warning(
                    f"LoRA adapter不存在（{self.lora_path}），"
                    f"合并模型也不存在（{self.merged_path}），降级为基座模型。"
                    f"如需微调，请先运行: python src/generation/finetune.py"
                )
                self._fallback_to_base(base_model)
        except Exception as e:
            logger.error(f"加载微调模型失败: {e}，降级为基座模型")
            self._fallback_to_base(base_model)

    def _load_merged_model(self):
        """加载合并后的完整模型"""
        logger.info(f"正在加载合并模型: {self.merged_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.merged_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.merged_path,
            torch_dtype=self.dtype,
            trust_remote_code=True,
            device_map="auto" if self.device == "cuda" else None,
        )
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        self.model.eval()
        self._is_finetuned = True
        logger.info("合并模型加载完成")

    def _load_lora_adapter(self):
        """加载基座模型 + LoRA adapter"""
        logger.info(f"正在加载基座模型 + LoRA adapter")
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=self.dtype,
            trust_remote_code=True,
            device_map="auto" if self.device == "cuda" else None,
        )
        if self.device == "cpu":
            base = base.to(self.device)
        self.model = PeftModel.from_pretrained(base, self.lora_path)
        self.model = self.model.merge_and_unload()
        self.model.eval()
        self._is_finetuned = True
        logger.info("LoRA adapter加载完成")

    def _fallback_to_base(self, base_model=None):
        """降级为基座模型"""
        if base_model is not None:
            self.model = base_model.model
            self.tokenizer = base_model.tokenizer
        else:
            # 自己加载基座模型
            from src.generation.base_model import BaseModelGenerator
            fallback = BaseModelGenerator()
            self.model = fallback.model
            self.tokenizer = fallback.tokenizer
        self._is_finetuned = False

    def generate(self, prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
        """
        使用微调模型生成回答。

        Args:
            prompt: 用户提示词
            system_prompt: 系统提示词

        Returns:
            模型回答
        """
        try:
            formatted = format_chatml_prompt(system_prompt, prompt)
            inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    repetition_penalty=REPETITION_PENALTY,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )

            response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            return response.strip()

        except Exception as e:
            logger.error(f"微调模型推理失败: {e}")
            return f"模型推理出错: {e}"
