"""
提示词模板模块

提供RAG增强问答和直接问答的提示词构建函数。
"""

from typing import List


def build_rag_prompt(query: str, retrieved_chunks: List[dict]) -> str:
    """
    将检索结果拼接到prompt中，构建RAG增强问答提示词。

    Args:
        query: 用户问题
        retrieved_chunks: 检索到的文档块列表，每项含 {"text", "source", "chunk_id", "score"}

    Returns:
        格式化后的完整prompt字符串
    """
    references = ""
    for i, chunk in enumerate(retrieved_chunks, start=1):
        source = chunk.get("source", "未知来源")
        text = chunk.get("text", "")
        references += f"\n[文档{i}，来源：{source}]\n{text}\n"

    prompt = f"""你是一个专业的领域知识助手。请根据以下参考资料回答用户问题。

【参考资料】
{references}
【用户问题】
{query}

【回答要求】
- 必须优先使用参考资料中的信息
- 如果参考资料不包含答案，请明确说"根据现有资料无法确定"
- 回答简洁专业，使用中文"""
    return prompt


def build_rag_prompt_with_history(query: str, retrieved_chunks: List[dict], history: List[dict]) -> str:
    """
    构建包含历史对话上下文的 RAG 增强问答提示词。

    Args:
        query: 当前用户问题
        retrieved_chunks: 检索到的文档块列表
        history: 历史对话消息 [{"role": "user"/"assistant", "content": str}, ...]

    Returns:
        格式化后的完整prompt字符串
    """
    references = ""
    for i, chunk in enumerate(retrieved_chunks, start=1):
        source = chunk.get("source", "未知来源")
        text = chunk.get("text", "")
        references += f"\n[文档{i}，来源：{source}]\n{text}\n"

    # 构建历史对话部分
    history_text = ""
    if history:
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            role_label = "用户" if role == "user" else "助手"
            history_text += f"{role_label}：{content}\n"

    prompt = f"""你是一个专业的领域知识助手。请根据以下参考资料和历史对话回答用户问题。

【参考资料】
{references}
【历史对话】
{history_text}
【当前用户问题】
{query}

【回答要求】
- 必须优先使用参考资料中的信息
- 结合历史对话上下文理解用户意图
- 如果参考资料不包含答案，请明确说"根据现有资料无法确定"
- 回答简洁专业，使用中文"""
    return prompt


def build_direct_prompt(query: str) -> str:
    """
    不使用RAG的直接问答prompt。

    Args:
        query: 用户问题

    Returns:
        格式化的prompt字符串
    """
    prompt = f"""你是一个专业的领域知识助手。请回答以下用户问题。

【用户问题】
{query}

【回答要求】
- 回答简洁专业，使用中文
- 如果不确定，请明确说明"""
    return prompt
