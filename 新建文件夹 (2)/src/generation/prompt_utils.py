"""
共享 Prompt 格式化工具

提供统一的对话模板格式化函数，供 base_model、fine_tuned_model 等模块共用。
"""
from config import SYSTEM_PROMPT


def format_chatml_prompt(system_prompt: str, user_prompt: str) -> str:
    """
    将 system/user 内容格式化为 ChatML 格式。

    适用于 DeepSeek-R1-Distill-Qwen、Qwen2.5 等 Qwen 系模型。

    Args:
        system_prompt: 系统提示词
        user_prompt: 用户提示词（含 RAG 参考资料等）

    Returns:
        ChatML 格式的完整 prompt 字符串
    """
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def format_chatml_messages(messages: list, system_prompt: str = SYSTEM_PROMPT) -> str:
    """
    将多轮对话消息列表格式化为 ChatML 格式。

    Args:
        messages: 消息列表，每项 {"role": "user"/"assistant", "content": str}
        system_prompt: 系统提示词

    Returns:
        ChatML 格式的完整 prompt 字符串（含历史对话）
    """
    parts = [f"<|im_start|>system\n{system_prompt}<|im_end|>"]
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)
