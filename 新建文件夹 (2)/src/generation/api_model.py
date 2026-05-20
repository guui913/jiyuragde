"""
DeepSeek API 调用模块（备选方案）

通过 DeepSeek API 生成回答，需要设置环境变量 DEEPSEEK_API_KEY。
"""

import logging
import requests

from config import DEEPSEEK_API_KEY, DEEPSEEK_API_BASE, DEEPSEEK_API_MODEL, SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class APIModelGenerator:
    """通过 DeepSeek API 生成回答"""

    def __init__(self, api_key: str = DEEPSEEK_API_KEY, model: str = DEEPSEEK_API_MODEL):
        """
        初始化API模型生成器。

        Args:
            api_key: DeepSeek API Key
            model: 模型名称
        """
        self.api_key = api_key
        self.api_base = DEEPSEEK_API_BASE
        self.model = model

        if not self.api_key:
            logger.warning(
                "DEEPSEEK_API_KEY 未设置。"
                "请设置环境变量: export DEEPSEEK_API_KEY=your_key"
            )
        else:
            logger.info(f"DeepSeek API 已配置，模型: {model}")

    def generate(self, prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
        """
        调用 DeepSeek API 生成回答。

        Args:
            prompt: 用户提示词
            system_prompt: 系统提示词

        Returns:
            API返回的回答
        """
        if not self.api_key:
            return "错误: DEEPSEEK_API_KEY 未设置，请设置环境变量后重试。"

        url = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 512,
            "top_p": 0.9,
        }

        try:
            logger.info("正在调用 DeepSeek API...")
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            data = response.json()
            answer = data["choices"][0]["message"]["content"]
            logger.info("DeepSeek API 调用成功")
            return answer.strip()

        except requests.exceptions.RequestException as e:
            logger.error(f"DeepSeek API 请求失败: {e}")
            return f"API调用出错: {e}"
        except (KeyError, IndexError) as e:
            logger.error(f"DeepSeek API 响应格式异常: {e}")
            return f"API响应解析出错: {e}"
