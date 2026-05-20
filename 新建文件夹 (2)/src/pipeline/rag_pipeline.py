"""
RAG 问答管道模块

编排完整的 RAG 问答流程：检索 → 构建prompt → 生成回答 → 返回结果。
支持多轮对话历史上下文。
"""

import logging
from typing import List, Optional

import chromadb

from config import FINAL_TOP_K, SYSTEM_PROMPT
from src.knowledge_base.load_kb import load_vector_store, get_chunks_metadata
from src.retrieval.hybrid_search import search as hybrid_retrieve
from src.retrieval.keyword_search import clear_bm25_cache
from src.pipeline.prompt_templates import build_rag_prompt, build_direct_prompt, build_rag_prompt_with_history

logger = logging.getLogger(__name__)


class RAGPipeline:
    """RAG 问答管道，编排检索和生成流程"""

    def __init__(self, model_type: str = "base"):
        """
        初始化 RAG Pipeline：加载检索器和生成器。

        Args:
            model_type: 模型类型
                - "base": 基座模型
                - "finetuned": 微调模型
                - "api": DeepSeek API

        Raises:
            ValueError: model_type 无效时抛出
            RuntimeError: 模型加载失败时抛出
            FileNotFoundError: 知识库不存在时抛出
        """
        self.model_type = model_type
        self.collection = None
        self.chunks_metadata = None
        self.generator = None

        # 加载生成器
        if model_type == "base":
            from src.generation.base_model import BaseModelGenerator
            logger.info("使用基座模型")
            self.generator = BaseModelGenerator()

        elif model_type == "finetuned":
            from src.generation.fine_tuned_model import FineTunedModelGenerator
            logger.info("使用微调模型")
            self.generator = FineTunedModelGenerator()

        elif model_type == "api":
            from src.generation.api_model import APIModelGenerator
            logger.info("使用 DeepSeek API")
            self.generator = APIModelGenerator()

        elif model_type == "ollama":
            from src.generation.ollama_model import OllamaModelGenerator
            logger.info("使用 Ollama 本地模型")
            self.generator = OllamaModelGenerator()

        else:
            raise ValueError(f"不支持的模型类型: '{model_type}'，可选: base, finetuned, api, ollama")

        logger.info(f"RAG Pipeline 初始化完成 (model_type={model_type})")

    def _ensure_knowledge_base(self):
        """确保知识库已加载（惰性加载）"""
        if self.collection is None:
            self.collection = load_vector_store()
        if self.chunks_metadata is None:
            self.chunks_metadata = get_chunks_metadata()

    def answer(self, query: str, history: Optional[List[dict]] = None) -> dict:
        """
        完整 RAG 问答流程：
        1. 混合检索
        2. 构建 RAG prompt（可选含历史对话）
        3. 模型生成回答
        4. 返回结果

        Args:
            query: 用户问题
            history: 可选，历史对话消息 [{"role": "user"/"assistant", "content": str}, ...]

        Returns:
            dict: {
                "query": str,
                "answer": str,
                "retrieved_chunks": List[dict],
                "model_type": str
            }
        """
        logger.info(f"RAG问答开始，query={query[:80]}...")

        try:
            # 1. 混合检索
            self._ensure_knowledge_base()
            retrieved_chunks = hybrid_retrieve(
                query,
                collection=self.collection,
                chunks_metadata=self.chunks_metadata,
                top_k=FINAL_TOP_K,
            )

            # 2. 构建 RAG prompt（含历史对话）
            if history:
                prompt = build_rag_prompt_with_history(query, retrieved_chunks, history)
            else:
                prompt = build_rag_prompt(query, retrieved_chunks)

            # 3. 生成回答
            answer_text = self.generator.generate(prompt)

            # 4. 构建返回结果
            result = {
                "query": query,
                "answer": answer_text,
                "retrieved_chunks": retrieved_chunks,
                "model_type": self.model_type,
            }

            logger.info(f"RAG问答完成，answer={answer_text[:80]}...")
            return result

        except FileNotFoundError as e:
            logger.warning(f"知识库不存在: {e}")
            # 降级为直接问答
            return self.answer_without_rag(query)

        except Exception as e:
            logger.error(f"RAG问答失败: {e}")
            return {
                "query": query,
                "answer": f"问答过程出错: {e}",
                "retrieved_chunks": [],
                "model_type": self.model_type,
            }

    def refresh_knowledge_base(self):
        """
        刷新知识库连接（文档上传后调用，同时清除 BM25 缓存）。
        """
        self.collection = None
        self.chunks_metadata = None
        clear_bm25_cache()
        logger.info("知识库连接已刷新，BM25 缓存已清除")

    def answer_without_rag(self, query: str) -> dict:
        """
        不使用 RAG，直接问答（用于对照实验）。

        Args:
            query: 用户问题

        Returns:
            dict: 同上格式，但 retrieved_chunks 为空
        """
        logger.info(f"直接问答开始，query={query[:80]}...")
        prompt = build_direct_prompt(query)
        answer_text = self.generator.generate(prompt)

        result = {
            "query": query,
            "answer": answer_text,
            "retrieved_chunks": [],
            "model_type": self.model_type,
        }
        logger.info(f"直接问答完成，answer={answer_text[:80]}...")
        return result
