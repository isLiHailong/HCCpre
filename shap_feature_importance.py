"""使用 SHAP 评估“仅 T1 输入”场景下各特征的贡献。

脚本流程：
1. 读取 processed_ML_data.csv 以及 5 折索引；
2. 在每折分别为 T2/T3 训练 HistGradientBoostingClassifier；
3. 以训练集子样本作为背景，借助 SHAP.Explainer 计算验证集上的 SHAP 值；
4. 输出“每折 × 每特征”的绝对值平均贡献以及跨折平均排名。

结果会写入 `/content/drive/.../feature_importance/shap/`，便于与 permutation 版本对照。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import HistGradientBoostingClassifier


PROJECT_ROOT = "/content/drive/MyDrive/Github项目/HCC预测机器学习"
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "processed_ML_data.csv")
FOLDS_DIR = os.path.join(PROJECT_ROOT, "folds")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "feature_importance")
SHAP_DIR = os.path.join(OUTPUT_ROOT, "shap")

os.makedirs(SHAP_DIR, exist_ok=True)


@dataclass
class ShapRecord:
    fold: int
    task: str
    feature: str
    mean_abs_shap: float


def read_smart(path: str) -> pd.DataFrame:
    for enc in ("utf-8-sig", "gb18030", "gbk", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception:
            continue
    return pd.read_csv(path, low_memory=False)


def read_indices(fold_dir: str, split: str) -> np.ndarray:
    csv_path = os.path.join(fold_dir, f"{split}.csv")
    df = read_smart(csv_path)
    if "idx" not in df.columns:
        raise RuntimeError(f"{csv_path} 缺少 idx 列，无法定位样本。")
    return df["idx"].to_numpy()


def ensure_binary(y: np.ndarray) -> bool:
    return np.unique(y).size >= 2


def select_t1_feature_cols(columns: list[str]) -> list[str]:
    return [c for c in columns if c.startswith("scaled_") and c.endswith("__T1")]


def build_explainer(model: HistGradientBoostingClassifier,
                    background: np.ndarray,
                    feature_names: list[str]) -> shap.Explainer:
    # SHAP 背景集不宜过大，默认采样最多 200 条
    sample_n = min(len(background), 200)
    if sample_n <= 0:
        raise RuntimeError("背景样本数量为 0，无法计算 SHAP。")
    bg = shap.utils.sample(background, sample_n, random_state=42)
    return shap.Explainer(model, bg, feature_names=feature_names)


def extract_class_values(explanation: shap.Explanation) -> np.ndarray:
    values = np.asarray(explanation.values)
    if values.ndim == 3:
        # 多分类情况下取阳性类别（索引 1），若只有一个类别则退回 0
        cls = 1 if values.shape[2] > 1 else 0
        values = values[..., cls]
    return values


def compute_fold_shap(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
) -> Tuple[np.ndarray, shap.Explanation]:
    model = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.05,
        max_iter=300,
        random_state=42,
    )
    model.fit(X_train, y_train)
    explainer = build_explainer(model, X_train, feature_names)
    explanation = explainer(X_val, check_additivity=False)
    shap_values = extract_class_values(explanation)
    return shap_values, explanation


def main() -> None:
    print("[Info] 加载 processed_ML_data.csv …")
    df = read_smart(PROCESSED_DATA_PATH)
    feature_cols = select_t1_feature_cols(df.columns.tolist())
    if not feature_cols:
        raise RuntimeError("processed_ML_data.csv 中没有 T1 对应的 scaled_ 特征列。")

    tasks = (("T2", "T2outcome"), ("T3", "T3outcome"))
    shap_records: list[ShapRecord] = []

    for fold in range(1, 6):
        fold_dir = os.path.join(FOLDS_DIR, f"fold_{fold}")
        if not os.path.isdir(fold_dir):
            print(f"[Warning] 找不到 {fold_dir}，跳过该折。")
            continue

        train_idx = read_indices(fold_dir, "train")
        val_idx = read_indices(fold_dir, "val")

        X_train = df.loc[train_idx, feature_cols].to_numpy()
        X_val = df.loc[val_idx, feature_cols].to_numpy()

        for task_name, label_col in tasks:
            y_train = df.loc[train_idx, label_col].to_numpy()
            y_val = df.loc[val_idx, label_col].to_numpy()

            if not ensure_binary(y_train) or not ensure_binary(y_val):
                print(f"[Warning] Fold {fold} {task_name} 标签单一，跳过 SHAP。")
                continue

            print(f"[Info] 计算 Fold {fold} {task_name} SHAP …")
            shap_values, _ = compute_fold_shap(X_train, X_val, y_train, feature_cols)
            mean_abs = np.mean(np.abs(shap_values), axis=0)

            for feature, score in zip(feature_cols, mean_abs):
                shap_records.append(
                    ShapRecord(
                        fold=fold,
                        task=task_name,
                        feature=feature.replace("scaled_", ""),
                        mean_abs_shap=float(score),
                    )
                )

    if not shap_records:
        raise RuntimeError("没有可用的 SHAP 结果，请检查 folds 与标签。")

    detail_df = pd.DataFrame([r.__dict__ for r in shap_records])
    summary_df = (
        detail_df.groupby(["task", "feature"], as_index=False)
        .agg(mean_abs_shap=("mean_abs_shap", "mean"))
        .sort_values(["task", "mean_abs_shap"], ascending=[True, False])
    )

    detail_path = os.path.join(SHAP_DIR, "shap_importance_by_fold.csv")
    summary_path = os.path.join(SHAP_DIR, "shap_feature_ranking_summary.csv")

    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"[Done] 每折 SHAP 重要性已保存到: {detail_path}")
    print(f"[Done] 跨折汇总已保存到: {summary_path}")

    for task_name in summary_df["task"].unique():
        top_df = summary_df.loc[summary_df["task"] == task_name].head(15)
        print(f"\n[Top 15] {task_name} 任务 SHAP 重要性：")
        for rank, row in enumerate(top_df.itertuples(index=False), 1):
            print(f" {rank:>2}. {row.feature:<25} mean_abs_shap={row.mean_abs_shap:.5f}")


if __name__ == "__main__":
    main()
