"""
对比实验运行模块

运行四组对比实验：
- Group A: 基座模型 direct 问答
- Group B: 微调模型 direct 问答
- Group C: 基座模型 + RAG
- Group D: 微调模型 + RAG
"""

import json
import logging
import os
from typing import List

from config import TRAINING_DIR, EVAL_RESULTS_DIR
from src.pipeline.rag_pipeline import RAGPipeline
from src.evaluation.auto_eval import compute_bleu, compute_rouge_l, load_eval_data

logger = logging.getLogger(__name__)


def run_single_experiment(pipeline: RAGPipeline, eval_data: List[dict], group_name: str, use_rag: bool = True) -> dict:
    """
    运行单组实验。

    Args:
        pipeline: RAGPipeline 实例
        eval_data: 评测数据
        group_name: 组名 (如 "Group_A")
        use_rag: 是否使用 RAG

    Returns:
        实验统计结果
    """
    logger.info(f"开始实验: {group_name} (use_rag={use_rag})")

    bleu_scores = []
    rouge_scores = []
    results = []

    for i, item in enumerate(eval_data):
        question = item.get("question", "")
        reference = item.get("reference_answer", "")

        if not question:
            continue

        try:
            if use_rag:
                pipeline_result = pipeline.answer(question)
            else:
                pipeline_result = pipeline.answer_without_rag(question)
            candidate = pipeline_result.get("answer", "")
        except Exception as e:
            logger.error(f"实验 {group_name} 第 {i+1} 条失败: {e}")
            candidate = ""

        bleu = compute_bleu(reference, candidate)
        rouge_l = compute_rouge_l(reference, candidate)

        bleu_scores.append(bleu)
        rouge_scores.append(rouge_l)

        results.append({
            "index": i + 1,
            "question": question,
            "reference_answer": reference,
            "candidate_answer": candidate,
            "bleu_4": bleu,
            "rouge_l": rouge_l,
        })

        if (i + 1) % 10 == 0:
            logger.info(f"{group_name} 进度: {i+1}/{len(eval_data)}")

    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0

    return {
        "group": group_name,
        "model_type": getattr(pipeline, "model_type", "unknown"),
        "use_rag": use_rag,
        "total": len(results),
        "avg_bleu_4": round(avg_bleu, 2),
        "avg_rouge_l": round(avg_rouge, 4),
        "results": results,
    }


def run_all_experiments(eval_data_path: str = None) -> dict:
    """
    运行四组对比实验：
    Group A: 纯基座模型 direct 问答
    Group B: 微调模型 direct 问答
    Group C: 基座模型 + RAG
    Group D: 微调模型 + RAG

    Args:
        eval_data_path: 评测数据路径

    Returns:
        完整实验结果 dict
    """
    if eval_data_path is None:
        eval_data_path = os.path.join(TRAINING_DIR, "eval.json")

    eval_data = load_eval_data(eval_data_path)

    all_results = []

    # Group A: 基座模型 + direct 问答
    logger.info("=" * 50)
    logger.info("Group A: 基座模型 direct 问答")
    logger.info("=" * 50)
    try:
        pipeline_a = RAGPipeline(model_type="base")
        result_a = run_single_experiment(pipeline_a, eval_data, "Group_A_Base_Direct", use_rag=False)
        all_results.append(result_a)
    except Exception as e:
        logger.error(f"Group A 实验失败: {e}")
        all_results.append({"group": "Group_A_Base_Direct", "error": str(e)})

    # Group B: 微调模型 + direct 问答
    logger.info("=" * 50)
    logger.info("Group B: 微调模型 direct 问答")
    logger.info("=" * 50)
    try:
        pipeline_b = RAGPipeline(model_type="finetuned")
        result_b = run_single_experiment(pipeline_b, eval_data, "Group_B_Finetuned_Direct", use_rag=False)
        all_results.append(result_b)
    except Exception as e:
        logger.error(f"Group B 实验失败: {e}")
        all_results.append({"group": "Group_B_Finetuned_Direct", "error": str(e)})

    # Group C: 基座模型 + RAG
    logger.info("=" * 50)
    logger.info("Group C: 基座模型 + RAG")
    logger.info("=" * 50)
    try:
        pipeline_c = RAGPipeline(model_type="base")
        result_c = run_single_experiment(pipeline_c, eval_data, "Group_C_Base_RAG", use_rag=True)
        all_results.append(result_c)
    except Exception as e:
        logger.error(f"Group C 实验失败: {e}")
        all_results.append({"group": "Group_C_Base_RAG", "error": str(e)})

    # Group D: 微调模型 + RAG
    logger.info("=" * 50)
    logger.info("Group D: 微调模型 + RAG")
    logger.info("=" * 50)
    try:
        pipeline_d = RAGPipeline(model_type="finetuned")
        result_d = run_single_experiment(pipeline_d, eval_data, "Group_D_Finetuned_RAG", use_rag=True)
        all_results.append(result_d)
    except Exception as e:
        logger.error(f"Group D 实验失败: {e}")
        all_results.append({"group": "Group_D_Finetuned_RAG", "error": str(e)})

    # 汇总
    summary = {
        "experiments": all_results,
        "summary": _generate_summary_table(all_results),
    }

    # 保存
    save_path = os.path.join(EVAL_RESULTS_DIR, "all_experiments_results.json")
    os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(f"四组实验全部完成，结果已保存: {save_path}")
    _print_summary(summary)

    return summary


def _generate_summary_table(all_results: list) -> List[dict]:
    """生成对比表格"""
    table = []
    for r in all_results:
        row = {
            "group": r.get("group", "unknown"),
            "avg_bleu_4": r.get("avg_bleu_4", "N/A"),
            "avg_rouge_l": r.get("avg_rouge_l", "N/A"),
        }
        table.append(row)
    return table


def _print_summary(summary: dict):
    """打印对比结果表格"""
    print("\n" + "=" * 60)
    print("对比实验结果汇总")
    print("=" * 60)
    print(f"{'实验组':<30} {'BLEU-4':>10} {'Rouge-L':>10}")
    print("-" * 60)
    for row in summary.get("summary", []):
        bleu = row.get("avg_bleu_4", "N/A")
        rouge = row.get("avg_rouge_l", "N/A")
        print(f"{row['group']:<30} {str(bleu):>10} {str(rouge):>10}")
    print("=" * 60)
