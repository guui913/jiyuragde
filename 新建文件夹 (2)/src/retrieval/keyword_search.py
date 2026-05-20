"""
关键词检索模块

基于 jieba 分词 + BM25 进行关键词级检索。
支持预缓存 BM25 索引以避免每次查询重建。
"""

import logging
from typing import List, Optional, Tuple

import jieba
from rank_bm25 import BM25Okapi

from config import HYBRID_TOP_K_KEYWORD

logger = logging.getLogger(__name__)

# BM25 索引缓存: (chunk_ids_tuple, BM25Okapi, tokenized_corpus)
_bm25_cache: Optional[Tuple[tuple, BM25Okapi, List[List[str]]]] = None


def _tokenize(text: str) -> List[str]:
    """
    使用 jieba 对文本进行分词。

    Args:
        text: 待分词文本

    Returns:
        分词结果列表
    """
    return list(jieba.cut(text))


def _get_or_build_bm25(corpus: List[str], chunk_ids: List[str]) -> BM25Okapi:
    """
    获取或构建 BM25 索引（带缓存）。
    如果 chunk_ids 列表与缓存一致则直接返回缓存的 BM25 实例。

    Args:
        corpus: 文档库文本列表
        chunk_ids: 对应的 chunk_id 列表

    Returns:
        BM25Okapi 实例
    """
    global _bm25_cache
    ids_key = tuple(chunk_ids)
    if _bm25_cache is not None and _bm25_cache[0] == ids_key:
        logger.debug("使用缓存的 BM25 索引")
        return _bm25_cache[1]

    # 重建索引
    logger.info(f"构建 BM25 索引，共 {len(chunk_ids)} 篇文档")
    tokenized_corpus = [_tokenize(doc) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    _bm25_cache = (ids_key, bm25, tokenized_corpus)
    return bm25


def clear_bm25_cache():
    """清除 BM25 缓存（知识库更新后调用）。"""
    global _bm25_cache
    _bm25_cache = None
    logger.info("BM25 缓存已清除")


def keyword_search(query: str, chunks: List[dict], top_k: int = HYBRID_TOP_K_KEYWORD) -> List[dict]:
    """
    基于关键词的检索，使用 jieba 分词和 BM25 算法。
    支持预缓存 BM25 索引以提高重复查询性能。

    Args:
        query: 用户问题
        chunks: 所有文档块列表（从metadata加载），每项含 text, source, chunk_id 等字段
        top_k: 返回结果数量

    Returns:
        List[dict]: 排序后的文档块列表，每项:
            {
                "text": str,
                "score": float,
                "source": str,
                "chunk_id": str
            }
    """
    if not chunks:
        logger.warning("文档块列表为空，无法进行关键词检索")
        return []

    logger.info(f"开始关键词检索，query={query[:50]}..., top_k={top_k}")

    # 提取所有文档块的文本和ID
    corpus = [chunk.get("text", "") for chunk in chunks]
    chunk_ids = [chunk.get("chunk_id", f"chunk_{i}") for i, chunk in enumerate(chunks)]

    # 获取缓存或新建 BM25 索引
    bm25 = _get_or_build_bm25(corpus, chunk_ids)

    # 对查询分词并计算得分
    tokenized_query = _tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    # 按得分降序排序，取 Top-K
    indexed_scores = list(enumerate(scores))
    indexed_scores.sort(key=lambda x: x[1], reverse=True)

    results = []
    for idx, score in indexed_scores[:top_k]:
        chunk = chunks[idx]
        results.append({
            "text": chunk.get("text", ""),
            "score": float(score),
            "source": chunk.get("source", "未知来源"),
            "chunk_id": chunk.get("chunk_id", f"chunk_{idx}"),
        })

    logger.info(f"关键词检索完成，返回 {len(results)} 条结果")
    return results
