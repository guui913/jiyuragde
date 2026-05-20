"""
混合检索模块

结合关键词检索和向量检索，通过 RRF (Reciprocal Rank Fusion) 算法进行融合排序。
"""

import logging
from typing import List

import chromadb

from config import FINAL_TOP_K, HYBRID_TOP_K_KEYWORD, HYBRID_TOP_K_VECTOR, RRF_K
from src.knowledge_base.load_kb import load_vector_store, get_chunks_metadata
from src.retrieval.keyword_search import keyword_search
from src.retrieval.vector_search import vector_search

logger = logging.getLogger(__name__)


def _rrf_score(rank_keyword: int, rank_vector: int, k: int = RRF_K) -> float:
    """
    计算 RRF 融合分数。

    Args:
        rank_keyword: 关键词检索中的排名（1-based）
        rank_vector: 向量检索中的排名（1-based）
        k: RRF常数

    Returns:
        RRF融合分数
    """
    return 1.0 / (k + rank_keyword) + 1.0 / (k + rank_vector)


def hybrid_search(
    query: str,
    keyword_results: List[dict],
    vector_results: List[dict],
    top_k: int = FINAL_TOP_K,
) -> List[dict]:
    """
    使用 RRF 算法融合关键词检索和向量检索结果。

    Args:
        query: 用户查询（用于日志）
        keyword_results: 关键词检索结果列表
        vector_results: 向量检索结果列表
        top_k: 最终返回结果数

    Returns:
        融合排序后的Top-K结果
    """
    logger.info(f"开始RRF混合检索融合，query={query[:50]}...")

    # 用 chunk_id 建立去重映射
    combined = {}  # chunk_id -> dict

    # 处理关键词检索结果（排名从1开始）
    for rank, item in enumerate(keyword_results, start=1):
        chunk_id = item.get("chunk_id", f"kw_{rank}")
        if chunk_id not in combined:
            combined[chunk_id] = {**item, "ranks": {}}
        combined[chunk_id]["ranks"]["keyword"] = rank

    # 处理向量检索结果
    for rank, item in enumerate(vector_results, start=1):
        chunk_id = item.get("chunk_id", f"vec_{rank}")
        if chunk_id not in combined:
            combined[chunk_id] = {**item, "ranks": {}}
        combined[chunk_id]["ranks"]["vector"] = rank

    # 计算 RRF 分数
    scored_results = []
    for chunk_id, item in combined.items():
        rank_kw = item["ranks"].get("keyword", 10000)  # 未出现的给一个大排名
        rank_vec = item["ranks"].get("vector", 10000)
        rrf = _rrf_score(rank_kw, rank_vec)
        item_copy = {k: v for k, v in item.items() if k != "ranks"}
        item_copy["score"] = rrf
        scored_results.append(item_copy)

    # 按 RRF 分数降序排序
    scored_results.sort(key=lambda x: x["score"], reverse=True)

    final = scored_results[:top_k]
    logger.info(f"RRF融合完成，最终返回 {len(final)} 条结果")
    return final


def search(
    query: str,
    collection: chromadb.Collection = None,
    chunks_metadata: List[dict] = None,
    top_k: int = FINAL_TOP_K,
) -> List[dict]:
    """
    一站式检索：关键词检索 + 向量检索 + RRF融合。

    Args:
        query: 用户查询
        collection: Chroma向量库collection（不传则自动加载）
        chunks_metadata: 分块元数据（不传则自动加载）
        top_k: 最终返回结果数

    Returns:
        融合后的检索结果
    """
    # 自动加载资源
    if collection is None:
        try:
            collection = load_vector_store()
        except Exception as e:
            logger.error(f"加载向量库失败: {e}")
            raise

    if chunks_metadata is None:
        try:
            chunks_metadata = get_chunks_metadata()
        except Exception as e:
            logger.error(f"加载分块元数据失败: {e}")
            raise

    # 关键词检索
    kw_results = keyword_search(query, chunks_metadata, top_k=HYBRID_TOP_K_KEYWORD)

    # 向量检索
    vec_results = vector_search(query, collection, top_k=HYBRID_TOP_K_VECTOR)

    # RRF 融合
    final_results = hybrid_search(query, kw_results, vec_results, top_k=top_k)

    return final_results
