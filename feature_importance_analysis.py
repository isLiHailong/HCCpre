"""计算仅使用 ``T1`` 时间点特征时的模型重要性排序。

该脚本严格沿用 Colab 工程固定路径：

1. 读取 `processed_ML_data.csv` 与 `folds/` 索引；
2. 在每个折上训练 `HistGradientBoostingClassifier`（分别对应 `T2`/`T3` 任务）；
3. 通过验证集的 permutation importance 量化每个特征对 ROC-AUC 的贡献；
4. 将每折结果与跨折平均排序写入 `feature_importance/` 目录，便于后续可解释性分析或特征筛选。

由于 permutation importance 使用验证集概率与真实标签计算 `roc_auc`，
它天然支持我们强调的 “训练吸收三次时间点，验证可单独看 T1” 的需求。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance


PROJECT_ROOT = "/content/drive/MyDrive/Github项目/HCC预测机器学习"
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "processed_ML_data.csv")
FOLDS_DIR = os.path.join(PROJECT_ROOT, "folds")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "feature_importance")

os.makedirs(OUTPUT_DIR, exist_ok=True)


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


def ensure_binary(labels: np.ndarray) -> bool:
    uniques = np.unique(labels)
    return uniques.size >= 2


@dataclass
class ImportanceResult:
    fold: int
    task: str
    feature: str
    importance_mean: float
    importance_std: float


TASKS = (
    ("T2", "T2outcome"),
    ("T3", "T3outcome"),
)


def select_t1_feature_cols(columns: list[str]) -> list[str]:
    """Return scaled feature列中只包含 ``__T1`` 后缀的部分。"""

    return [c for c in columns if c.startswith("scaled_") and c.endswith("__T1")]


def train_and_rank(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = select_t1_feature_cols(df.columns.tolist())
    if not feature_cols:
        raise RuntimeError("processed_ML_data.csv 中找不到 T1 对应的 scaled_ 特征列。")

    raw_records: list[ImportanceResult] = []

    for fold in range(1, 6):
        fold_dir = os.path.join(FOLDS_DIR, f"fold_{fold}")
        if not os.path.isdir(fold_dir):
            print(f"[Warning] 缺少 {fold_dir}，跳过该折。")
            continue

        train_idx = read_indices(fold_dir, "train")
        val_idx = read_indices(fold_dir, "val")

        X_train = df.loc[train_idx, feature_cols].to_numpy()
        X_val = df.loc[val_idx, feature_cols].to_numpy()

        for task_name, label_col in TASKS:
            y_train = df.loc[train_idx, label_col].to_numpy()
            y_val = df.loc[val_idx, label_col].to_numpy()

            if not ensure_binary(y_train) or not ensure_binary(y_val):
                print(
                    f"[Warning] Fold {fold} {task_name} 标签单一，跳过 permutation importance。"
                )
                continue

            clf = HistGradientBoostingClassifier(
                max_depth=6,
                learning_rate=0.05,
                max_iter=300,
                random_state=42,
            )
            clf.fit(X_train, y_train)

            perm = permutation_importance(
                clf,
                X_val,
                y_val,
                scoring="roc_auc",
                n_repeats=10,
                random_state=42,
                n_jobs=-1,
            )

            for feature, mean_imp, std_imp in zip(
                feature_cols, perm.importances_mean, perm.importances_std
            ):
                raw_records.append(
                    ImportanceResult(
                        fold=fold,
                        task=task_name,
                        feature=feature.replace("scaled_", ""),
                        importance_mean=float(mean_imp),
                        importance_std=float(std_imp),
                    )
                )

    if not raw_records:
        raise RuntimeError("没有可用的特征重要性结果，请确认 folds 与标签是否齐全。")

    raw_df = pd.DataFrame([r.__dict__ for r in raw_records])
    summary_df = (
        raw_df.groupby(["task", "feature"], as_index=False)
        .agg(
            mean_importance=("importance_mean", "mean"),
            std_importance=("importance_mean", "std"),
            mean_std=("importance_std", "mean"),
        )
        .sort_values(["task", "mean_importance"], ascending=[True, False])
    )

    return raw_df, summary_df


def main() -> None:
    print("[Info] 加载 processed_ML_data.csv …")
    df = read_smart(PROCESSED_DATA_PATH)

    raw_df, summary_df = train_and_rank(df)

    raw_path = os.path.join(OUTPUT_DIR, "permutation_importance_by_fold.csv")
    summary_path = os.path.join(OUTPUT_DIR, "feature_ranking_summary.csv")

    raw_df.to_csv(raw_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"[Done] 每折重要性已保存到: {raw_path}")
    print(f"[Done] 汇总排名已保存到: {summary_path}")

    for task_name in summary_df["task"].unique():
        top_df = summary_df.loc[summary_df["task"] == task_name].head(15)
        print(f"\n[Top 15] {task_name} 任务最重要的特征：")
        for rank, row in enumerate(top_df.itertuples(index=False), 1):
            print(
                f" {rank:>2}. {row.feature:<25}  mean={row.mean_importance:.4f}  std={row.std_importance:.4f}"
            )


if __name__ == "__main__":
    main()
