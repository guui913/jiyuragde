"""
向量检索模块

基于 Chroma 向量数据库进行语义检索。
"""

import logging
from typing import List

import chromadb

from config import HYBRID_TOP_K_VECTOR
from src.knowledge_base.embedding import get_embedding_model

logger = logging.getLogger(__name__)


def vector_search(query: str, collection: chromadb.Collection, top_k: int = HYBRID_TOP_K_VECTOR) -> List[dict]:
    """
    基于向量嵌入的语义检索。

    Args:
        query: 用户查询
        collection: Chroma向量库collection
        top_k: 返回结果数量

    Returns:
        List[dict]: 检索结果，每项:
            {
                "text": str,
                "score": float,
                "source": str,
                "chunk_id": str
            }
    """
    logger.info(f"开始向量检索，query={query[:50]}..., top_k={top_k}")

    try:
        model = get_embedding_model()
        # 将查询向量化
        query_embedding = model.encode(query, normalize_embeddings=True).tolist()

        # 查询 Chroma
        raw_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        results = []
        if raw_results and raw_results.get("documents") and raw_results["documents"][0]:
            for i in range(len(raw_results["documents"][0])):
                doc = raw_results["documents"][0][i] if raw_results["documents"] else ""
                meta = raw_results["metadatas"][0][i] if raw_results["metadatas"] else {}
                chunk_id = raw_results["ids"][0][i] if raw_results["ids"] else f"vec_{i}"

                # 距离转相似度：如果返回 distance，转换为余弦相似度
                distance = raw_results.get("distances", [[1.0]])[0][i] if raw_results.get("distances") else 1.0
                score = float(1.0 - distance)  # 转为相似度

                results.append({
                    "text": doc,
                    "score": score,
                    "source": meta.get("source", "未知来源"),
                    "chunk_id": chunk_id,
                })

        logger.info(f"向量检索完成，返回 {len(results)} 条结果")
        return results

    except Exception as e:
        logger.error(f"向量检索失败: {e}")
        return []
