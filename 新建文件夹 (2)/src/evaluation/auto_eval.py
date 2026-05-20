"""
自动评估模块

使用 BLEU-4 和 Rouge-L 对模型输出进行自动评估。
"""

import json
import logging
import os
from typing import List

from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer

from config import TRAINING_DIR, EVAL_RESULTS_DIR

logger = logging.getLogger(__name__)


def compute_bleu(reference: str, candidate: str) -> float:
    """
    计算 BLEU-4 分数。

    Args:
        reference: 参考答案
        candidate: 模型生成的候选答案

    Returns:
        BLEU-4 分数 (0-100)
    """
    if not candidate or not reference:
        return 0.0
    try:
        bleu = corpus_bleu([candidate], [[reference]])
        return bleu.score
    except Exception as e:
        logger.error(f"计算BLEU失败: {e}")
        return 0.0


def compute_rouge_l(reference: str, candidate: str) -> float:
    """
    计算 Rouge-L F1 分数。

    Args:
        reference: 参考答案
        candidate: 模型生成的候选答案

    Returns:
        Rouge-L F1 分数 (0-1)
    """
    if not candidate or not reference:
        return 0.0
    try:
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        scores = scorer.score(reference, candidate)
        return scores["rougeL"].fmeasure
    except Exception as e:
        logger.error(f"计算Rouge-L失败: {e}")
        return 0.0


def load_eval_data(eval_data_path: str) -> List[dict]:
    """
    加载评测数据集。

    Args:
        eval_data_path: eval.json 路径

    Returns:
        评测数据列表
    """
    if not os.path.exists(eval_data_path):
        # 尝试拼接默认路径
        default_path = os.path.join(TRAINING_DIR, "eval.json")
        if os.path.exists(default_path):
            eval_data_path = default_path
        else:
            raise FileNotFoundError(
                f"评测数据文件不存在: {eval_data_path} 或 {default_path}"
            )

    with open(eval_data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"加载评测数据: {eval_data_path}，共 {len(data)} 条")
    return data


def evaluate_pipeline(pipeline, eval_data_path: str = None) -> dict:
    """
    对评测集中每个问题，调用 pipeline 获取回答并计算 BLEU-4 和 Rouge-L。

    Args:
        pipeline: RAGPipeline 实例
        eval_data_path: 评测数据路径

    Returns:
        dict: {
            "results": List[dict],   # 每条评测的详细结果
            "avg_bleu": float,
            "avg_rouge_l": float,
            "total": int,
            "model_type": str,
        }
    """
    if eval_data_path is None:
        eval_data_path = os.path.join(TRAINING_DIR, "eval.json")

    eval_data = load_eval_data(eval_data_path)

    results = []
    bleu_scores = []
    rouge_scores = []

    logger.info(f"开始自动评估，共 {len(eval_data)} 条评测数据")

    for i, item in enumerate(eval_data):
        question = item.get("question", "")
        reference = item.get("reference_answer", "")

        if not question:
            logger.warning(f"第 {i+1} 条数据缺少 question，跳过")
            continue

        # 调用 pipeline 获取回答
        try:
            pipeline_result = pipeline.answer(question)
            candidate = pipeline_result.get("answer", "")
        except Exception as e:
            logger.error(f"第 {i+1} 条数据问答失败: {e}")
            candidate = ""

        # 计算指标
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
            logger.info(f"评估进度: {i+1}/{len(eval_data)}")

    # 计算统计结果
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    avg_rouge_l = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0

    summary = {
        "results": results,
        "avg_bleu": round(avg_bleu, 2),
        "avg_rouge_l": round(avg_rouge_l, 4),
        "total": len(results),
        "model_type": getattr(pipeline, "model_type", "unknown"),
    }

    # 保存结果
    save_path = os.path.join(EVAL_RESULTS_DIR, "auto_eval_results.json")
    os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(f"自动评估完成，平均BLEU-4: {avg_bleu:.2f}, 平均Rouge-L: {avg_rouge_l:.4f}")
    logger.info(f"评估结果已保存: {save_path}")

    return summary
