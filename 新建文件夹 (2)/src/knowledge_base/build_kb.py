"""
知识库构建模块

读取 data/raw/ 下的所有 .txt/.pdf 文件，进行文档分块、向量化，并存入 Chroma 向量库。
"""

import json
import logging
import os
import shutil
from typing import List

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    RAW_DATA_DIR,
    PROCESSED_DIR,
    CHROMA_PERSIST_DIR,
    CHUNKS_JSON_PATH,
    CHROMA_COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from src.knowledge_base.embedding import get_embedding_model

logger = logging.getLogger(__name__)


def _read_pdf(file_path: str) -> str:
    """
    读取 PDF 文件的文本内容。

    Args:
        file_path: PDF 文件路径

    Returns:
        提取的文本内容
    """
    # 优先使用 pymupdf（fitz），其次 pdfplumber，最后 PyPDF2
    try:
        import fitz
        doc = fitz.open(file_path)
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
        return "\n".join(text_parts)
    except ImportError:
        pass

    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text_parts.append(t)
        return "\n".join(text_parts)
    except ImportError:
        pass

    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(file_path)
        text_parts = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text_parts.append(t)
        return "\n".join(text_parts)
    except ImportError:
        raise ImportError(
            "读取 PDF 需要安装 PDF 解析库。请运行: pip install pymupdf  (推荐) "
            "或 pip install pdfplumber  或 pip install PyPDF2"
        )


def _read_documents(raw_dir: str) -> List[dict]:
    """
    读取 raw 目录下的所有 .txt 和 .pdf 文件。

    Args:
        raw_dir: 原始文档目录

    Returns:
        List[dict]: 文档列表，每项 {"source": str, "content": str}
    """
    documents = []
    if not os.path.exists(raw_dir):
        logger.warning(f"原始文档目录不存在: {raw_dir}")
        return documents

    supported_exts = {".txt", ".pdf"}
    files = [f for f in os.listdir(raw_dir)
             if os.path.splitext(f)[1].lower() in supported_exts]
    if not files:
        logger.warning(f"data/raw/ 目录下没有支持的文档文件 (.txt / .pdf)")
        return documents

    for filename in files:
        file_path = os.path.join(raw_dir, filename)
        ext = os.path.splitext(filename)[1].lower()
        try:
            if ext == ".pdf":
                content = _read_pdf(file_path)
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            if content.strip():
                documents.append({
                    "source": filename,
                    "content": content,
                })
                logger.info(f"读取文档: {filename} ({len(content)} 字符)")
            else:
                logger.warning(f"文档为空，跳过: {filename}")
        except ImportError as e:
            logger.error(str(e))
            print(f"跳过 {filename}: {e}")
        except Exception as e:
            logger.error(f"读取文档失败 {filename}: {e}")

    return documents


def _split_documents(documents: List[dict]) -> List[dict]:
    """
    使用 RecursiveCharacterTextSplitter 对文档分块。

    Args:
        documents: 文档列表

    Returns:
        List[dict]: 分块列表，每项 {"chunk_id", "source", "chunk_index", "text", "char_count"}
    """
    logger.info(f"开始文档分块 (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],
    )

    all_chunks = []
    for doc in documents:
        source = doc["source"]
        # 用 source 名作为 chunk_id 前缀
        base_name = os.path.splitext(source)[0]
        chunks = splitter.split_text(doc["content"])

        for i, chunk_text in enumerate(chunks, start=1):
            chunk_id = f"{base_name}_{i:03d}"
            all_chunks.append({
                "chunk_id": chunk_id,
                "source": source,
                "chunk_index": i,
                "text": chunk_text,
                "char_count": len(chunk_text),
            })

    logger.info(f"文档分块完成，共 {len(all_chunks)} 个块")
    return all_chunks


def _save_chunks_metadata(chunks: List[dict]) -> None:
    """
    保存分块元数据到 JSON 文件。

    Args:
        chunks: 分块列表
    """
    try:
        os.makedirs(os.path.dirname(CHUNKS_JSON_PATH), exist_ok=True)
        with open(CHUNKS_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        logger.info(f"分块元数据已保存: {CHUNKS_JSON_PATH}")
    except Exception as e:
        logger.error(f"保存分块元数据失败: {e}")
        raise


def _build_vector_store(chunks: List[dict]) -> chromadb.Collection:
    """
    向量化文档块并存入 Chroma。

    Args:
        chunks: 分块列表

    Returns:
        Chroma collection 对象
    """
    logger.info(f"开始向量化并存入 Chroma，共 {len(chunks)} 个块")

    # 创建 Chroma 客户端
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    # 如果 collection 已存在则删除重建
    try:
        client.delete_collection(name=CHROMA_COLLECTION_NAME)
        logger.info(f"已删除旧 collection: {CHROMA_COLLECTION_NAME}")
    except Exception:
        pass

    collection = client.create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # 加载嵌入模型
    model = get_embedding_model()

    # 批量向量化并存入 (batch_size=32)
    batch_size = 32
    texts = [chunk["text"] for chunk in chunks]
    ids = [chunk["chunk_id"] for chunk in chunks]
    metadatas = [
        {"source": chunk["source"], "chunk_index": chunk["chunk_index"], "char_count": chunk["char_count"]}
        for chunk in chunks
    ]

    total_batches = (len(texts) + batch_size - 1) // batch_size
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        batch_metas = metadatas[i:i + batch_size]

        # 向量化
        embeddings = model.encode(batch_texts, normalize_embeddings=True).tolist()

        # 存入 Chroma
        collection.add(
            ids=batch_ids,
            embeddings=embeddings,
            documents=batch_texts,
            metadatas=batch_metas,
        )

        batch_num = i // batch_size + 1
        logger.info(f"向量化进度: {batch_num}/{total_batches} ({min(i+batch_size, len(texts))}/{len(texts)})")

    logger.info(f"向量库构建完成，已存入 {collection.count()} 条记录")
    return collection


def _add_chunks_incremental(chunks: List[dict]) -> bool:
    """
    增量添加文档块到已有向量库（不重建）。

    Args:
        chunks: 新的分块列表

    Returns:
        是否成功
    """
    if not chunks:
        return True

    logger.info(f"增量添加 {len(chunks)} 个块到向量库")
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    try:
        collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
    except Exception:
        # collection 不存在，需要先创建
        collection = client.create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # 过滤掉已存在的 chunk_id
    existing_ids = set()
    try:
        existing = collection.get(include=[])
        if existing and existing.get("ids"):
            existing_ids = set(existing["ids"])
    except Exception:
        pass

    new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]
    if not new_chunks:
        logger.info("所有分块已存在，跳过")
        return True

    model = get_embedding_model()
    batch_size = 32
    texts = [c["text"] for c in new_chunks]
    ids = [c["chunk_id"] for c in new_chunks]
    metadatas = [
        {"source": c["source"], "chunk_index": c["chunk_index"], "char_count": c["char_count"]}
        for c in new_chunks
    ]

    total_batches = (len(texts) + batch_size - 1) // batch_size
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        batch_metas = metadatas[i:i + batch_size]
        embeddings = model.encode(batch_texts, normalize_embeddings=True).tolist()
        collection.add(
            ids=batch_ids,
            embeddings=embeddings,
            documents=batch_texts,
            metadatas=batch_metas,
        )
        batch_num = i // batch_size + 1
        logger.info(f"增量向量化进度: {batch_num}/{total_batches}")

    logger.info(f"增量添加完成，新增 {len(new_chunks)} 条")
    return True


def build_knowledge_base() -> bool:
    """
    构建知识库完整流程：读取文档 → 分块 → 向量化 → 存入Chroma。
    同时返回是否成功。

    Returns:
        bool: 构建是否成功
    """
    try:
        # 1. 检查raw目录
        if not os.path.exists(RAW_DATA_DIR):
            logger.error(f"原始文档目录不存在: {RAW_DATA_DIR}")
            print(f"错误: 目录不存在 {RAW_DATA_DIR}")
            return False

        # 2. 读取文档
        documents = _read_documents(RAW_DATA_DIR)
        if not documents:
            print("请先将领域文档（.txt格式）放入 data/raw/ 目录")
            return False

        logger.info(f"共读取 {len(documents)} 个文档")

        # 3. 文档分块
        chunks = _split_documents(documents)

        # 4. 保存分块元数据
        _save_chunks_metadata(chunks)

        # 5. 构建向量库
        _build_vector_store(chunks)

        print(f"知识库构建完成！共处理 {len(documents)} 个文档，生成 {len(chunks)} 个文本块。")
        return True

    except Exception as e:
        logger.error(f"知识库构建失败: {e}")
        print(f"知识库构建失败: {e}")
        return False


def add_document(file_path: str) -> bool:
    """
    增量添加单个文档到知识库（不重建已有数据）。

    Args:
        file_path: 文档路径

    Returns:
        是否成功
    """
    try:
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            return False

        filename = os.path.basename(file_path)
        # 复制文件到 raw 目录
        dest = os.path.join(RAW_DATA_DIR, filename)
        shutil.copy2(file_path, dest)
        logger.info(f"文档已复制到 {dest}")

        # 读取新文档内容
        ext = os.path.splitext(filename)[1].lower()
        if ext == ".pdf":
            content = _read_pdf(dest)
        else:
            with open(dest, "r", encoding="utf-8") as f:
                content = f.read()

        if not content.strip():
            logger.warning(f"文档为空: {filename}")
            return False

        new_doc = [{"source": filename, "content": content}]

        # 分块
        new_chunks = _split_documents(new_doc)
        logger.info(f"新文档分块: {len(new_chunks)} 个块")

        # 更新元数据 JSON（合并）
        existing_metadata = []
        if os.path.exists(CHUNKS_JSON_PATH):
            with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
                existing_metadata = json.load(f)
        existing_metadata.extend(new_chunks)
        _save_chunks_metadata(existing_metadata)

        # 增量添加向量
        _add_chunks_incremental(new_chunks)

        print(f"文档已入库: {filename} (新增 {len(new_chunks)} 个文本块)")
        return True

    except Exception as e:
        logger.error(f"添加文档失败: {e}")
        return False


def rebuild_knowledge_base() -> bool:
    """
    完全重建知识库（清空并重新构建）。
    适用于 --build-kb 命令行参数。

    Returns:
        bool: 构建是否成功
    """
    return build_knowledge_base()
