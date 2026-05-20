"""
Gradio 前端界面模块 - 现代化设计

领域智能问答系统的 Web 交互界面。
支持多轮对话上下文、新对话、对话管理等功能。
"""

import html
import json
import logging
import os
import time
from typing import List, Optional

import gradio as gr

from src.pipeline.rag_pipeline import RAGPipeline
from src.knowledge_base.build_kb import add_document

logger = logging.getLogger(__name__)

# 模型名称映射
MODEL_DISPLAY_NAMES = {
    "ollama": "Ollama (本地)",
    "api": "DeepSeek API",
    "base": "基座模型 (本地)",
    "finetuned": "微调模型",
}


def _build_status_bar(model_type: str, last_time: Optional[float] = None) -> str:
    """动态构建状态栏 HTML"""
    display_name = MODEL_DISPLAY_NAMES.get(model_type, model_type)
    time_part = ""
    if last_time is not None and last_time > 0:
        time_part = f'<span style="color:#64748b;">|</span><span>⏱️ 上次响应: <b>{last_time:.2f}s</b></span>'
    return (
        '<div class="status-bar">'
        '<span><span class="status-dot"></span>系统就绪</span>'
        '<span style="color:#64748b;">|</span>'
        f'<span>当前模型: <b>{display_name}</b></span>'
        '<span style="color:#64748b;">|</span>'
        '<span>知识库: <b>已加载</b></span>'
        f'{time_part}'
        '</div>'
    )


def _format_sources_markdown(retrieved_chunks: List[dict]) -> str:
    """将检索到的文档块格式化为精美的来源面板内容"""
    if not retrieved_chunks:
        return (
            '<div style="text-align:center; padding:40px 20px; color:#94a3b8;">'
            '<div style="font-size:48px; margin-bottom:12px;">📭</div>'
            '<div style="font-size:15px; font-weight:500;">暂无检索来源</div>'
            '<div style="font-size:13px; margin-top:4px;">提问后将在此显示相关参考资料</div>'
            '</div>'
        )

    html_parts = ['<div style="display:flex; flex-direction:column; gap:10px;">']
    for i, chunk in enumerate(retrieved_chunks, start=1):
        source = html.escape(chunk.get("source", "未知来源"))
        score = chunk.get("score", 0)
        text = chunk.get("text", "")
        chunk_id = html.escape(chunk.get("chunk_id", f"chunk_{i}"))
        preview = text[:180].replace("\n", " ") + "…" if len(text) > 180 else text.replace("\n", " ")
        escaped_preview = html.escape(preview)

        # 相关度颜色指示
        if score > 0.5:
            badge_color = "#16a34a"
            badge_bg = "#dcfce7"
            badge_text = "🔥 高相关"
        elif score > 0.1:
            badge_color = "#ca8a04"
            badge_bg = "#fef9c3"
            badge_text = "📌 中相关"
        else:
            badge_color = "#6b7280"
            badge_bg = "#f3f4f6"
            badge_text = "📎 低相关"

        html_parts.append(f'''
        <div style="background:#fff; border-radius:10px; padding:14px 16px;
                    border:1px solid #e5e7eb; transition:all 0.2s;">
            <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:8px;">
                <span style="font-size:13px; font-weight:600; color:#1e293b;">
                    <span style="color:#3b82f6;">#{i}</span> 📄 {source}
                </span>
                <span style="font-size:11px; padding:2px 10px; border-radius:12px;
                             color:{badge_color}; background:{badge_bg}; font-weight:600;">
                    {badge_text}
                </span>
            </div>
            <div style="font-size:13px; color:#475569; line-height:1.6;">
                {escaped_preview}
            </div>
            <div style="display:flex; justify-content:space-between; margin-top:6px;
                        font-size:11px; color:#94a3b8;">
                <span>ID: {chunk_id}</span>
                <span>Score: {score:.3f}</span>
            </div>
        </div>''')

    html_parts.append('</div>')
    return ''.join(html_parts)


CUSTOM_CSS = """
    /* 全局 */
    body, .gradio-container {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans SC", sans-serif;
    }

    /* 主标题 */
    .main-title {
        text-align: center;
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 50%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 32px !important;
        font-weight: 800 !important;
        margin-bottom: 4px !important;
        letter-spacing: -0.5px;
    }
    .sub-title {
        text-align: center;
        color: #64748b;
        font-size: 14px;
        margin-top: -4px;
        margin-bottom: 20px;
    }

    /* 状态栏 */
    .status-bar {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 20px;
        padding: 6px 20px;
        background: #f0f9ff;
        border-radius: 10px;
        margin-bottom: 16px;
        font-size: 13px;
        color: #1e40af;
    }
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #22c55e;
        display: inline-block;
        margin-right: 6px;
    }

    /* 聊天区 */
    .chat-column > div {
        border-radius: 12px !important;
        border: 1px solid #e5e7eb !important;
        background: #ffffff !important;
    }

    /* 来源面板 */
    .sources-column > div {
        border-radius: 12px !important;
        border: 1px solid #e5e7eb !important;
        background: #f8fafc !important;
        padding: 8px !important;
        min-height: 500px;
        max-height: 560px;
        overflow-y: auto;
    }

    /* 输入框 */
    .query-box textarea {
        border-radius: 12px !important;
        border: 2px solid #e2e8f0 !important;
        padding: 14px 18px !important;
        font-size: 15px !important;
        transition: border-color 0.3s !important;
        background: #ffffff !important;
    }
    .query-box textarea:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59,130,246,0.1) !important;
    }

    /* 按钮 */
    .btn-submit {
        background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        height: 52px !important;
        transition: all 0.3s !important;
    }
    .btn-submit:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(37,99,235,0.3) !important;
    }

    /* 底部控制栏 */
    .control-row {
        background: #f8fafc;
        border-radius: 12px;
        padding: 12px 16px;
        margin-top: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
    }

    /* 下拉框 */
    .model-select select {
        border-radius: 8px !important;
        background: #fff !important;
        border: 1px solid #e2e8f0 !important;
        font-weight: 500 !important;
        font-size: 13px !important;
    }

    /* 对话列表 */
    .conv-list {
        background: #f8fafc;
        border-radius: 12px;
        padding: 8px;
        border: 1px solid #e5e7eb;
        min-height: 200px;
        max-height: 560px;
        overflow-y: auto;
    }
    .conv-item {
        background: #fff;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 6px;
        border: 1px solid #e2e8f0;
        cursor: pointer;
        transition: all 0.2s;
        font-size: 13px;
    }
    .conv-item:hover {
        border-color: #3b82f6;
        background: #eff6ff;
    }
    .conv-item.active {
        border-color: #3b82f6;
        background: #dbeafe;
        font-weight: 600;
    }
    .btn-new-conv {
        background: linear-gradient(135deg, #10b981, #059669) !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        color: white !important;
        transition: all 0.3s !important;
    }
    .btn-new-conv:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(16,185,129,0.3) !important;
    }
    .app-footer {
        text-align: center;
        color: #94a3b8;
        font-size: 12px;
        padding: 4px;
        margin-top: 6px;
    }
    /* 示例问题 */
    .example-chip {
        background: #fff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        font-size: 12px !important;
        color: #475569 !important;
        transition: all 0.2s !important;
        text-align: left !important;
        width: 100% !important;
        min-height: 32px !important;
        padding: 6px 12px !important;
        margin-bottom: 4px !important;
    }
    .example-chip:hover {
        border-color: #3b82f6 !important;
        background: #eff6ff !important;
        color: #1e40af !important;
    }
    /* 高级参数面板 */
    .param-row {
        background: #f8fafc;
        border-radius: 12px;
        padding: 12px 20px;
        margin-top: 8px;
    }
    /* 知识库概览 */
    .kb-stats-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
        padding: 8px;
    }
    .kb-stat-box {
        background: #fff;
        border-radius: 10px;
        padding: 16px 12px;
        text-align: center;
        border: 1px solid #e5e7eb;
    }
    .kb-stat-num {
        font-size: 24px;
        font-weight: 700;
        color: #1e40af;
    }
    .kb-stat-label {
        font-size: 11px;
        color: #64748b;
        margin-top: 4px;
    }
    .kb-source-item {
        background: #fff;
        border-radius: 6px;
        padding: 6px 10px;
        margin: 4px 8px;
        font-size: 12px;
        color: #475569;
        border: 1px solid #e5e7eb;
    }
    """


def create_ui(pipeline: RAGPipeline):
    """创建并返回 Gradio Blocks 界面"""

    def _get_kb_stats() -> dict:
        """获取知识库统计信息"""
        try:
            from src.knowledge_base.load_kb import load_vector_store, get_chunks_metadata
            from config import RAW_DATA_DIR
            chunks = get_chunks_metadata()
            collection = load_vector_store()
            total_vectors = collection.count() if collection else 0
            sources = set()
            total_chars = 0
            for c in chunks:
                sources.add(c.get("source", ""))
                total_chars += c.get("char_count", 0)
            return {
                "loaded": True,
                "document_count": len(sources),
                "chunk_count": len(chunks),
                "vector_count": total_vectors,
                "total_chars": total_chars,
                "sources": sorted(sources),
            }
        except Exception:
            return {"loaded": False}

    def _build_kb_dashboard_html() -> str:
        """构建知识库概览面板 HTML"""
        stats = _get_kb_stats()
        if not stats.get("loaded"):
            return (
                '<div style="text-align:center; padding:50px 20px; color:#f59e0b;">'
                '<div style="font-size:48px; margin-bottom:12px;">📭</div>'
                '<div style="font-size:15px; font-weight:600;">知识库未构建</div>'
                '<div style="font-size:13px; margin-top:8px;">请先运行: python run.py --build-kb</div>'
                '</div>'
            )
        sources_html = ""
        for s in stats.get("sources", []):
            escaped = html.escape(s)
            sources_html += f'<div class="kb-source-item">📄 {escaped}</div>'
        return f'''
        <div style="padding:4px;">
            <div class="kb-stats-grid">
                <div class="kb-stat-box">
                    <div class="kb-stat-num">{stats["document_count"]}</div>
                    <div class="kb-stat-label">源文档</div>
                </div>
                <div class="kb-stat-box">
                    <div class="kb-stat-num">{stats["chunk_count"]}</div>
                    <div class="kb-stat-label">文本块</div>
                </div>
                <div class="kb-stat-box">
                    <div class="kb-stat-num">{stats["vector_count"]}</div>
                    <div class="kb-stat-label">向量索引</div>
                </div>
                <div class="kb-stat-box">
                    <div class="kb-stat-num">{stats["total_chars"] // 1000:,}K</div>
                    <div class="kb-stat-label">总字符</div>
                </div>
            </div>
            <div style="font-size:13px; font-weight:600; color:#1e293b; padding:8px 10px 4px;">
                📂 文档清单 ({len(stats.get("sources", []))})
            </div>
            {sources_html}
        </div>'''

    with gr.Blocks(title="领域智能问答系统") as demo:
        # === 状态 ===
        current_conv_id = gr.State("conv_0")
        # conversations: {conv_id: {"title": str, "messages": list, "created_at": float}}
        conversations_state = gr.State({})

        # 标题 + 统计卡片
        stats = _get_kb_stats()
        stats_cards_html = ""
        if stats.get("loaded"):
            stats_cards_html = (
                '<div style="display:flex; justify-content:center; gap:24px; margin-bottom:10px;">'
                f'<div style="text-align:center;"><span style="font-size:24px;font-weight:700;color:#1e40af;">{stats["document_count"]}</span><br><span style="font-size:12px;color:#64748b;">📄 文档</span></div>'
                f'<div style="text-align:center;"><span style="font-size:24px;font-weight:700;color:#059669;">{stats["chunk_count"]}</span><br><span style="font-size:12px;color:#64748b;">🧩 分块</span></div>'
                f'<div style="text-align:center;"><span style="font-size:24px;font-weight:700;color:#7c3aed;">{stats["vector_count"]}</span><br><span style="font-size:12px;color:#64748b;">🔢 向量</span></div>'
                f'<div style="text-align:center;"><span style="font-size:24px;font-weight:700;color:#0891b2;">{stats["total_chars"] // 1000}K</span><br><span style="font-size:12px;color:#64748b;">📊 字符</span></div>'
                '</div>'
            )
        gr.HTML(
            '<div style="text-align:center;">'
            '<h1 class="main-title">领域智能问答系统</h1>'
            '<p class="sub-title">基于 RAG 检索增强生成 · 专业知识智能助手</p>'
            + stats_cards_html +
            '</div>'
        )

        # 状态栏（动态）
        last_response_time = gr.State(None)
        status_bar = gr.HTML(value=_build_status_bar(pipeline.model_type))

        # 主布局：对话侧边栏 + 聊天区 + 检索来源
        with gr.Row(equal_height=True):
            # 左侧对话侧边栏
            with gr.Column(scale=1, min_width=200):
                gr.HTML(
                    '<div style="font-size:14px; font-weight:600; color:#1e293b; '
                    'padding:4px 0 8px 0;">💬 对话列表</div>'
                )
                conversation_panel = gr.HTML(
                    value='<div class="conv-list">'
                          '<div style="text-align:center; padding:30px; color:#94a3b8; '
                          'font-size:13px;">暂无对话</div>'
                          '</div>'
                )
                new_conv_btn = gr.Button(
                    "✨ 新对话",
                    elem_classes=["btn-new-conv"],
                    size="sm",
                )

                # === 示例问题 ===
                gr.HTML(
                    '<div style="font-size:13px; font-weight:600; color:#1e293b; '
                    'padding:16px 0 6px 0;">💡 试试这些问题</div>'
                )
                example_questions = [
                    "劳动法中加班费怎么计算？",
                    "劳动合同解除有哪些法定情形？",
                    "试用期的期限最长为多久？",
                    "工伤认定标准是什么？",
                    "竞业限制的期限和补偿标准？",
                ]
                example_btns = []
                for eq in example_questions:
                    btn = gr.Button(
                        eq[:24] + ("…" if len(eq) > 24 else ""),
                        size="sm",
                        elem_classes=["example-chip"],
                    )
                    example_btns.append(btn)

            # 中间对话区
            with gr.Column(scale=3, elem_classes=["chat-column"]):
                chatbot = gr.Chatbot(
                    label="对话",
                    height=520,
                )

            # 右侧面板 - Tab切换
            with gr.Column(scale=2, elem_classes=["sources-column"]):
                with gr.Tabs():
                    with gr.TabItem("📚 检索来源"):
                        sources_panel = gr.HTML(
                            value=_format_sources_markdown([]),
                        )
                    with gr.TabItem("📊 知识库概览"):
                        kb_dashboard = gr.HTML(
                            value=_build_kb_dashboard_html(),
                        )

        # 输入区
        with gr.Row():
            with gr.Column(scale=8, elem_classes=["query-box"]):
                query_input = gr.Textbox(
                    placeholder="💬  输入您的问题，例如：劳动法中加班费怎么算？",
                    show_label=False,
                    lines=2,
                )
            with gr.Column(scale=1, min_width=90):
                submit_btn = gr.Button(
                    "🚀 提 问",
                    variant="primary",
                    elem_classes=["btn-submit"],
                )

        # 控制栏
        with gr.Row(elem_classes=["control-row"]):
            model_dropdown = gr.Dropdown(
                choices=["ollama", "api", "base", "finetuned"],
                value=pipeline.model_type,
                label="🧠 模型选择",
                scale=3,
                interactive=True,
            )
            upload_btn = gr.UploadButton(
                "📤 上传文档",
                file_types=[".txt", ".pdf"],
                scale=2,
                size="sm",
            )
            clear_btn = gr.Button(
                "🗑️ 清空对话",
                scale=2,
                size="sm",
            )
            export_btn = gr.Button(
                "📥 导出对话",
                scale=2,
                size="sm",
            )

        # === 高级参数面板 ===
        with gr.Accordion("⚙️ 高级生成参数", open=False, elem_classes=["param-row"]):
            with gr.Row():
                temp_slider = gr.Slider(
                    minimum=0.0, maximum=2.0, value=0.3, step=0.05,
                    label="🌡️ Temperature", info="越高越随机"
                )
                top_p_slider = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.9, step=0.05,
                    label="📊 Top-P", info="核采样阈值"
                )
                max_tokens_slider = gr.Slider(
                    minimum=64, maximum=2048, value=512, step=64,
                    label="✏️ Max Tokens", info="最大生成长度"
                )

        # 底部
        gr.HTML(
            '<div class="app-footer">'
            '基于 DeepSeek / Ollama · BGE 嵌入 · Chroma 向量库 · RRF 混合检索'
            '</div>'
        )

        # === 交互逻辑 ===

        def _get_history_for_pipeline(chatbot_history: list) -> list:
            """
            从 Chatbot 的历史消息中提取对话历史（不含当前问题），
            用于传递给 RAGPipeline 作为上下文。
            返回格式: [{"role": "user"/"assistant", "content": str}, ...]
            """
            if not chatbot_history:
                return []
            # chatbot_history 格式: [{"role": ..., "content": ...}, ...]
            return [{"role": m["role"], "content": m["content"]} for m in chatbot_history]

        def _render_conversation_list(convs: dict, active_id: str) -> str:
            """渲染对话列表 HTML"""
            if not convs:
                return (
                    '<div class="conv-list">'
                    '<div style="text-align:center; padding:30px; color:#94a3b8; '
                    'font-size:13px;">暂无对话</div>'
                    '</div>'
                )
            items_html = ""
            for cid in reversed(list(convs.keys())):
                conv = convs[cid]
                title = conv.get("title", "新对话")
                msg_count = len(conv.get("messages", []))
                active_class = " active" if cid == active_id else ""
                items_html += (
                    f'<div class="conv-item{active_class}" data-conv-id="{cid}">'
                    f'<div style="font-weight:600; margin-bottom:2px;">{title}</div>'
                    f'<div style="font-size:11px; color:#94a3b8;">{msg_count} 条消息</div>'
                    f'</div>'
                )
            return f'<div class="conv-list">{items_html}</div>'

        def respond(message: str, history: list, convs: dict, active_id: str,
                    temperature: float = 0.3, top_p: float = 0.9, max_tokens: int = 512) -> tuple:
            """处理用户提问（支持高级参数）"""
            if not message or not message.strip():
                return (history, _format_sources_markdown([]), convs, active_id,
                        _render_conversation_list(convs, active_id), None,
                        _build_status_bar(pipeline.model_type), _build_kb_dashboard_html())

            history_context = _get_history_for_pipeline(history)

            t0 = time.time()
            try:
                result = pipeline.answer(message, history=history_context)
                answer = result.get("answer", "无法生成回答")
                chunks = result.get("retrieved_chunks", [])
            except Exception as e:
                answer = f"问答出错: {e}"
                chunks = []
            elapsed = time.time() - t0

            # 在回答末尾追加耗时信息
            display_answer = answer + f"\n\n<small style='color:#94a3b8;'>⏱️ {elapsed:.2f}s · {len(answer)} 字</small>"
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": display_answer})

            if active_id in convs:
                convs[active_id]["messages"] = history
                if len(history) <= 2:
                    convs[active_id]["title"] = html.escape(message[:30]) + ("…" if len(message) > 30 else "")

            sources_html = _format_sources_markdown(chunks)
            conv_list_html = _render_conversation_list(convs, active_id)
            status_html = _build_status_bar(pipeline.model_type, elapsed)
            kb_html = _build_kb_dashboard_html()
            return history, sources_html, convs, active_id, conv_list_html, elapsed, status_html, kb_html

        def new_conversation(convs: dict, active_id: str) -> tuple:
            """创建新对话"""
            # 生成新对话 ID
            new_id = f"conv_{int(time.time() * 1000)}"
            convs[new_id] = {
                "title": "新对话",
                "messages": [],
                "created_at": time.time(),
            }
            conv_list_html = _render_conversation_list(convs, new_id)
            return [], _format_sources_markdown([]), convs, new_id, conv_list_html

        def clear_chat(convs: dict, active_id: str) -> tuple:
            """清空当前对话"""
            if active_id in convs:
                convs[active_id]["messages"] = []
                convs[active_id]["title"] = "新对话"
            conv_list_html = _render_conversation_list(convs, active_id)
            return [], _format_sources_markdown([]), convs, active_id, conv_list_html

        def handle_upload(file, convs: dict, active_id: str):
            """处理文档上传"""
            if file is None:
                return _format_sources_markdown([]), convs, active_id, _render_conversation_list(convs, active_id)
            file_path = file.name if hasattr(file, "name") else str(file)
            try:
                success = add_document(file_path)
                if success:
                    # 刷新管道中的知识库连接
                    try:
                        pipeline.refresh_knowledge_base()
                    except Exception:
                        pass
                    result_html = (
                        '<div style="text-align:center; padding:30px; color:#16a34a;">'
                        '<div style="font-size:36px;">✅</div>'
                        '<div style="font-size:14px; font-weight:600; margin-top:8px;">文档已入库</div>'
                        f'<div style="font-size:12px; color:#64748b; margin-top:4px;">{file_path}</div>'
                        '</div>'
                    )
                else:
                    result_html = '<div style="text-align:center; padding:30px; color:#dc2626;">❌ 添加失败</div>'
            except Exception as e:
                result_html = f'<div style="text-align:center; padding:30px; color:#dc2626;">❌ 错误: {e}</div>'
            conv_list_html = _render_conversation_list(convs, active_id)
            return result_html, convs, active_id, conv_list_html

        def switch_model(model_type: str):
            """切换模型"""
            try:
                pipeline.model_type = model_type
                if model_type == "base":
                    from src.generation.base_model import BaseModelGenerator
                    pipeline.generator = BaseModelGenerator()
                elif model_type == "finetuned":
                    from src.generation.fine_tuned_model import FineTunedModelGenerator
                    pipeline.generator = FineTunedModelGenerator()
                elif model_type == "api":
                    from src.generation.api_model import APIModelGenerator
                    pipeline.generator = APIModelGenerator()
                elif model_type == "ollama":
                    from src.generation.ollama_model import OllamaModelGenerator
                    pipeline.generator = OllamaModelGenerator()

                status_html = _build_status_bar(model_type)
                return _format_sources_markdown([]), status_html
            except Exception as e:
                return _format_sources_markdown([]), f'<div style="color:#dc2626;">切换失败: {e}</div>'

        def export_conversation(convs: dict, active_id: str) -> str:
            """导出当前对话为 Markdown 文件"""
            if active_id not in convs or not convs[active_id].get("messages"):
                return "⚠️ 当前对话为空，无可导出内容。"
            conv = convs[active_id]
            title = conv.get("title", "对话记录")
            lines = [f"# {title}", f"*导出: {time.strftime('%Y-%m-%d %H:%M:%S')}*", ""]
            for msg in conv["messages"]:
                role = "🧑 用户" if msg["role"] == "user" else "🤖 助手"
                content = msg["content"]
                # 移除末尾的耗时标签
                if content.endswith("</small>") and "⏱️" in content:
                    idx = content.rfind("\n\n<small")
                    if idx > 0:
                        content = content[:idx]
                lines.append(f"**{role}**：{content}")
                lines.append("")
            md_text = "\n".join(lines)
            export_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "data", "exports")
            os.makedirs(export_dir, exist_ok=True)
            filename = f"conversation_{time.strftime('%Y%m%d_%H%M%S')}.md"
            filepath = os.path.join(export_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(md_text)
            return f"✅ 已导出到 {filepath}\n\n---\n\n{md_text[:300]}…"

        def fill_example(text: str):
            """填充示例问题到输入框"""
            return text

        def refresh_kb_panels():
            """刷新知识库面板"""
            return _build_kb_dashboard_html()

        # === 绑定事件 ===

        # 提问按钮
        msg = submit_btn.click(
            fn=respond,
            inputs=[query_input, chatbot, conversations_state, current_conv_id, temp_slider, top_p_slider, max_tokens_slider],
            outputs=[chatbot, sources_panel, conversations_state, current_conv_id, conversation_panel, last_response_time, status_bar, kb_dashboard],
        )
        msg.then(lambda: "", outputs=[query_input])

        # 回车提交
        msg_enter = query_input.submit(
            fn=respond,
            inputs=[query_input, chatbot, conversations_state, current_conv_id, temp_slider, top_p_slider, max_tokens_slider],
            outputs=[chatbot, sources_panel, conversations_state, current_conv_id, conversation_panel, last_response_time, status_bar, kb_dashboard],
        )
        msg_enter.then(lambda: "", outputs=[query_input])

        # 示例问题点击 → 填入并自动提交
        for i, btn in enumerate(example_btns):
            def _make_fill(val):
                return lambda: val
            e1 = btn.click(fn=_make_fill(example_questions[i]), outputs=[query_input], queue=False)
            e1.then(
                fn=respond,
                inputs=[query_input, chatbot, conversations_state, current_conv_id, temp_slider, top_p_slider, max_tokens_slider],
                outputs=[chatbot, sources_panel, conversations_state, current_conv_id, conversation_panel, last_response_time, status_bar, kb_dashboard],
            ).then(lambda: "", outputs=[query_input])

        # 新对话
        new_conv_btn.click(
            fn=new_conversation,
            inputs=[conversations_state, current_conv_id],
            outputs=[chatbot, sources_panel, conversations_state, current_conv_id, conversation_panel],
        )

        # 清空对话
        clear_btn.click(
            fn=clear_chat,
            inputs=[conversations_state, current_conv_id],
            outputs=[chatbot, sources_panel, conversations_state, current_conv_id, conversation_panel],
        )

        # 导出对话
        export_btn.click(
            fn=export_conversation,
            inputs=[conversations_state, current_conv_id],
            outputs=[query_input],
        )

        # 上传文档
        upload_btn.upload(
            fn=handle_upload,
            inputs=[upload_btn, conversations_state, current_conv_id],
            outputs=[sources_panel, conversations_state, current_conv_id, conversation_panel],
        ).then(
            fn=refresh_kb_panels,
            outputs=[kb_dashboard],
        )

        # 模型切换
        model_dropdown.change(
            fn=switch_model,
            inputs=[model_dropdown],
            outputs=[sources_panel, status_bar],
        )

    return demo


def launch_ui(pipeline: RAGPipeline, share: bool = False):
    """启动 Gradio 界面"""
    demo = create_ui(pipeline)
    logger.info("正在启动 Gradio 界面...")
    print("\n" + "=" * 50)
    print("  领域智能问答系统 启动中...")
    print("  请在浏览器中打开上方显示的地址")
    print("=" * 50 + "\n")
    demo.launch(share=share, server_name="0.0.0.0", css=CUSTOM_CSS, theme=gr.themes.Soft())

