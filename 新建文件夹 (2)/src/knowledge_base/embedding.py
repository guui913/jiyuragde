"""
共享嵌入模型模块

提供全局嵌入模型单例，供 knowledge_base 和 retrieval 子包共用。
"""
import logging
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)

_embedding_model = None


def get_embedding_model() -> SentenceTransformer:
    """
    获取嵌入模型单例（惰性加载，线程安全注意：Gradio/Flask 多线程环境下 SentenceTransformer 自身线程安全）。

    Returns:
        SentenceTransformer 模型实例
    """
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"正在加载嵌入模型: {EMBEDDING_MODEL_NAME}")
        try:
            _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            logger.info("嵌入模型加载完成")
        except Exception as e:
            logger.error(f"嵌入模型加载失败: {e}")
            raise RuntimeError(f"无法加载嵌入模型 {EMBEDDING_MODEL_NAME}: {e}")
    return _embedding_model
