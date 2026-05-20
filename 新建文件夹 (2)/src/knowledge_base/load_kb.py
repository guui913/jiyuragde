"""
知识库加载模块

加载已构建的Chroma向量库和分块元数据。
"""

import json
import logging
from typing import List

import chromadb

from config import CHROMA_PERSIST_DIR, CHUNKS_JSON_PATH, CHROMA_COLLECTION_NAME, EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)


def load_vector_store() -> chromadb.Collection:
    """
    加载向量库，返回Chroma collection。

    Returns:
        chromadb.Collection: Chroma向量集合对象

    Raises:
        FileNotFoundError: Chroma持久化目录不存在
        RuntimeError: Chroma加载失败
    """
    try:
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        # 检查 collection 是否存在
        collection_names = client.list_collections()
        name_list = [c.name for c in collection_names]

        if CHROMA_COLLECTION_NAME not in name_list:
            raise RuntimeError(
                f"向量库 '{CHROMA_COLLECTION_NAME}' 不存在。"
                f"请先运行: python run.py --build-kb"
            )

        collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
        logger.info(f"成功加载向量库: {CHROMA_COLLECTION_NAME}")
        return collection

    except FileNotFoundError:
        logger.error(f"Chroma持久化目录不存在: {CHROMA_PERSIST_DIR}")
        raise FileNotFoundError(
            f"未找到知识库目录 {CHROMA_PERSIST_DIR}。"
            f"请先运行: python run.py --build-kb"
        )
    except Exception as e:
        logger.error(f"加载向量库失败: {e}")
        raise RuntimeError(f"加载向量库失败: {e}")


def get_chunks_metadata() -> List[dict]:
    """
    从JSON文件加载分块元数据。

    Returns:
        List[dict]: 分块元数据列表，每项含 chunk_id, source, chunk_index, text, char_count
    """
    try:
        with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        logger.info(f"成功加载分块元数据，共 {len(metadata)} 条")
        return metadata
    except FileNotFoundError:
        logger.error(f"分块元数据文件不存在: {CHUNKS_JSON_PATH}")
        raise FileNotFoundError(
            f"未找到分块元数据 {CHUNKS_JSON_PATH}。"
            f"请先运行: python run.py --build-kb"
        )
    except json.JSONDecodeError as e:
        logger.error(f"解析分块元数据JSON失败: {e}")
        raise RuntimeError(f"解析分块元数据失败: {e}")
