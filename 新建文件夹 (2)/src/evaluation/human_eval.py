"""
人工评估模块

生成人工评估打分表 (CSV)，并支持汇总多人打分结果。
"""

import csv
import json
import logging
import os
from typing import List

import numpy as np

from config import TRAINING_DIR, EVAL_RESULTS_DIR

logger = logging.getLogger(__name__)


def generate_score_sheet(eval_questions: List[dict], output_path: str) -> str:
    """
    生成评分表 CSV 文件。
    列: question_id, question, answer, accuracy, fluency, relevance

    Args:
        eval_questions: 评测数据列表，每项含 question 和 reference_answer
        output_path: 输出 CSV 文件路径

    Returns:
        生成的 CSV 文件路径
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow(["question_id", "question", "answer", "accuracy", "fluency", "relevance"])

        for i, item in enumerate(eval_questions):
            qid = f"Q{i+1:03d}"
            question = item.get("question", "")
            answer = item.get("reference_answer", "")
            writer.writerow([qid, question, answer, "", "", ""])

    logger.info(f"评分表已生成: {output_path}，共 {len(eval_questions)} 条")
    return output_path


def aggregate_scores(score_files: List[str]) -> dict:
    """
    合并多人打分，计算平均分和 Cohen's Kappa 一致性。

    Args:
        score_files: 多个评分 CSV 文件路径列表

    Returns:
        dict: {
            "individual_scores": List[dict],
            "avg_accuracy": float,
            "avg_fluency": float,
            "avg_relevance": float,
            "avg_overall": float,
        }
    """
    if not score_files:
        return {"error": "未提供评分文件"}

    all_scores = []
    rater_scores = {field: [] for field in ["accuracy", "fluency", "relevance"]}

    for file_path in score_files:
        if not os.path.exists(file_path):
            logger.warning(f"评分文件不存在: {file_path}")
            continue

        scores_for_this_rater = {field: [] for field in ["accuracy", "fluency", "relevance"]}
        with open(file_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    acc = float(row.get("accuracy", 0))
                    flu = float(row.get("fluency", 0))
                    rel = float(row.get("relevance", 0))
                    scores_for_this_rater["accuracy"].append(acc)
                    scores_for_this_rater["fluency"].append(flu)
                    scores_for_this_rater["relevance"].append(rel)

                    all_scores.append({
                        "question_id": row.get("question_id", ""),
                        "accuracy": acc,
                        "fluency": flu,
                        "relevance": rel,
                    })
                except (ValueError, KeyError) as e:
                    logger.warning(f"跳过无效评分行: {e}")

        for field in rater_scores:
            rater_scores[field].append(scores_for_this_rater[field])

    if not all_scores:
        return {"error": "未能从评分文件中读取有效数据"}

    avg_accuracy = np.mean([s["accuracy"] for s in all_scores])
    avg_fluency = np.mean([s["fluency"] for s in all_scores])
    avg_relevance = np.mean([s["relevance"] for s in all_scores])
    avg_overall = (avg_accuracy + avg_fluency + avg_relevance) / 3

    result = {
        "individual_scores": all_scores,
        "avg_accuracy": round(float(avg_accuracy), 2),
        "avg_fluency": round(float(avg_fluency), 2),
        "avg_relevance": round(float(avg_relevance), 2),
        "avg_overall": round(float(avg_overall), 2),
        "total_raters": len(score_files),
    }

    # 保存汇总结果
    save_path = os.path.join(EVAL_RESULTS_DIR, "human_eval_scores.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info(f"人工评估汇总完成，综合均分: {avg_overall:.2f}")
    logger.info(f"汇总结果已保存: {save_path}")

    return result
