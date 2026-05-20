"""
基座模型加载与推理模块

加载 DeepSeek-R1-Distill-Qwen-1.5B 基座模型，
自动检测CUDA可用性，使用合适的精度加载。
"""

import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import BASE_MODEL_NAME, SYSTEM_PROMPT, TEMPERATURE, TOP_P, MAX_NEW_TOKENS, REPETITION_PENALTY
from src.generation.prompt_utils import format_chatml_prompt, format_chatml_messages

logger = logging.getLogger(__name__)


class BaseModelGenerator:
    """基座模型生成器，加载并推理基座模型"""

    def __init__(self, model_name: str = BASE_MODEL_NAME):
        """
        加载基座模型和分词器。

        Args:
            model_name: HuggingFace模型名称或本地路径
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.device == "cuda":
            logger.info("检测到GPU，使用CUDA推理")
            self.dtype = torch.float16
        else:
            logger.warning("未检测到GPU，使用CPU推理（速度较慢）")
            self.dtype = torch.float32

        try:
            logger.info(f"正在加载基座模型: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=self.dtype,
                trust_remote_code=True,
                device_map="auto" if self.device == "cuda" else None,
            )
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            self.model.eval()
            logger.info("基座模型加载完成")
        except Exception as e:
            logger.error(f"基座模型加载失败: {e}")
            raise RuntimeError(f"无法加载基座模型 {model_name}: {e}")

    def _format_chat_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """
        使用对话模板格式化输入。

        Args:
            system_prompt: 系统提示词
            user_prompt: 用户提示词

        Returns:
            格式化后的输入文本
        """
        return format_chatml_prompt(system_prompt, user_prompt)

    def _format_chat_with_history(self, messages: list, system_prompt: str = SYSTEM_PROMPT) -> str:
        """
        将多轮对话历史格式化为模型输入。

        Args:
            messages: 消息列表 [{"role", "content"}, ...]
            system_prompt: 系统提示词

        Returns:
            格式化后的输入文本
        """
        return format_chatml_messages(messages, system_prompt)

    def generate(self, prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
        """
        输入prompt，返回模型回答。

        Args:
            prompt: 用户提示词（已拼接好的完整问题）
            system_prompt: 系统提示词

        Returns:
            模型生成的回答文本
        """
        try:
            formatted_input = self._format_chat_prompt(system_prompt, prompt)
            inputs = self.tokenizer(formatted_input, return_tensors="pt").to(self.device)

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

            # 解码输出，移除输入部分
            response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            return response.strip()

        except Exception as e:
            logger.error(f"模型推理失败: {e}")
            return f"模型推理出错: {e}"
