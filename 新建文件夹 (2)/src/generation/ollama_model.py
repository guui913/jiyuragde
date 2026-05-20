"""
Ollama 本地模型调用模块

通过 Ollama REST API 调用本地部署的大模型。
前提：已安装 Ollama 并已拉取模型，如 ollama pull deepseek-r1:1.5b
"""

import logging
import requests

from config import OLLAMA_BASE_URL, OLLAMA_MODEL_NAME, SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class OllamaModelGenerator:
    """通过 Ollama 本地 API 生成回答"""

    def __init__(self, model: str = OLLAMA_MODEL_NAME, base_url: str = OLLAMA_BASE_URL):
        """
        初始化 Ollama 模型生成器。

        Args:
            model: Ollama 模型名称，如 "deepseek-r1:1.5b", "qwen2.5:1.5b"
            base_url: Ollama 服务地址
        """
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate(self, prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
        """
        调用 Ollama API 生成回答。

        Args:
            prompt: 用户提示词
            system_prompt: 系统提示词

        Returns:
            模型回答
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "top_p": 0.9,
                "num_predict": 512,
            },
        }

        try:
            logger.info(f"正在调用 Ollama ({self.model})...")
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            answer = data.get("response", "")
            logger.info("Ollama 调用成功")
            return answer.strip()

        except requests.exceptions.ConnectionError:
            logger.error("无法连接到 Ollama，请确保 ollama serve 已启动")
            return "错误: 无法连接 Ollama 服务。请先运行 'ollama serve' 启动服务。"
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama 请求失败: {e}")
            return f"Ollama 调用出错: {e}"
        except Exception as e:
            logger.error(f"Ollama 响应解析失败: {e}")
            return f"Ollama 响应异常: {e}"
