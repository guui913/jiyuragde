"""
领域智能问答系统 - 启动入口

用法：
  python run.py                   # 启动Gradio界面（默认基座模型+RAG）
  python run.py --model finetuned # 使用微调模型
  python run.py --model api       # 使用DeepSeek API
  python run.py --build-kb        # 构建知识库
  python run.py --eval            # 运行评估实验
  python run.py --eval-all        # 运行四组对比实验
"""

import argparse
import logging
import sys

from config import EVAL_RESULTS_DIR

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="领域智能问答系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run.py                    # 启动Gradio界面
  python run.py --model finetuned  # 使用微调模型
  python run.py --model api        # 使用DeepSeek API
  python run.py --build-kb         # 构建知识库
  python run.py --eval             # 运行自动评估
  python run.py --eval-all         # 运行四组对比实验
        """,
    )
    parser.add_argument(
        "--model",
        choices=["base", "finetuned", "api", "ollama"],
        default="base",
        help="选择模型类型 (默认: base)",
    )
    parser.add_argument(
        "--build-kb",
        action="store_true",
        help="构建/重建知识库",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="运行自动评估（需要 data/training/eval.json）",
    )
    parser.add_argument(
        "--eval-all",
        action="store_true",
        help="运行四组对比实验（需要 data/training/eval.json）",
    )

    args = parser.parse_args()

    # 构建知识库
    if args.build_kb:
        from src.knowledge_base.build_kb import build_knowledge_base
        print("正在构建知识库...")
        success = build_knowledge_base()
        if success:
            print("知识库构建成功！")
            sys.exit(0)
        else:
            print("知识库构建失败。")
            sys.exit(1)

    # 运行自动评估
    elif args.eval:
        from src.pipeline.rag_pipeline import RAGPipeline
        from src.evaluation.auto_eval import evaluate_pipeline

        print(f"正在初始化 Pipeline (model_type={args.model})...")
        pipeline = RAGPipeline(model_type=args.model)

        print("正在运行自动评估...")
        results = evaluate_pipeline(pipeline)
        print(f"\n评估完成！")
        print(f"  评测条数: {results.get('total', 0)}")
        print(f"  平均 BLEU-4: {results.get('avg_bleu', 'N/A')}")
        print(f"  平均 Rouge-L: {results.get('avg_rouge_l', 'N/A')}")
        print(f"  结果已保存到: {EVAL_RESULTS_DIR}/auto_eval_results.json")
        sys.exit(0)

    # 运行四组对比实验
    elif args.eval_all:
        from src.evaluation.run_experiments import run_all_experiments

        print("正在运行四组对比实验...")
        print("此过程可能需要较长时间，请耐心等待。\n")
        summary = run_all_experiments()
        print(f"\n全部实验完成！")
        print(f"  结果已保存到: {EVAL_RESULTS_DIR}/all_experiments_results.json")
        sys.exit(0)

    # 默认：启动 Gradio 界面
    else:
        from src.pipeline.rag_pipeline import RAGPipeline
        from src.ui.gradio_app import launch_ui

        print(f"正在初始化 RAG Pipeline (model_type={args.model})...")
        try:
            pipeline = RAGPipeline(model_type=args.model)
        except Exception as e:
            print(f"初始化失败: {e}")
            print("\n提示:")
            print(f"  - 如果知识库未构建，请先运行: python run.py --build-kb")
            print(f"  - 如果模型下载失败，请设置环境变量: HF_ENDPOINT=https://hf-mirror.com")
            sys.exit(1)

        print(f"\n模型类型: {args.model}")
        launch_ui(pipeline, share=False)


if __name__ == "__main__":
    main()
