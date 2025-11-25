import os
import warnings
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

warnings.filterwarnings("ignore")

PROJECT_ROOT = "/content/drive/MyDrive/Github项目/HCC预测机器学习"
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "processed_ML_data.csv")
FOLDS_DIR = os.path.join(PROJECT_ROOT, "folds")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "单特征逻辑回归")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def read_smart(path: str) -> pd.DataFrame:
    for enc in ["utf-8-sig", "gb18030", "gbk", "utf-8"]:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception:
            continue
    return pd.read_csv(path, low_memory=False)


def find_best_threshold_youden(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden = tpr - fpr
    best_idx = np.argmax(youden)
    th = thresholds[best_idx]
    if th < 0.0:
        th = 0.0
    if th > 1.0:
        th = 1.0
    return th


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    try:
        metrics["AUC"] = roc_auc_score(y_true, y_prob)
    except Exception:
        metrics["AUC"] = np.nan
    try:
        metrics["PR_AUC"] = average_precision_score(y_true, y_prob)
    except Exception:
        metrics["PR_AUC"] = np.nan

    y_pred = (y_prob >= threshold).astype(int)
    metrics["Accuracy"] = accuracy_score(y_true, y_pred)
    metrics["Balanced_Accuracy"] = balanced_accuracy_score(y_true, y_pred)
    metrics["Sensitivity"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["Specificity"] = (
        confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()[0:2].sum()
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["Specificity"] = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    metrics["PPV"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["NPV"] = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    metrics["F1"] = f1_score(y_true, y_pred, zero_division=0)
    metrics["MCC"] = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_pred)) > 1 else np.nan
    try:
        metrics["Brier"] = brier_score_loss(y_true, y_prob)
    except Exception:
        metrics["Brier"] = np.nan
    return metrics


TIME_SUFFIXES = ("__T1", "__T2", "__T3")


def collect_feature_sets(df: pd.DataFrame) -> List[List[str]]:
    """Group columns by基准名，包含 T1/T2/T3 三个时间点（若存在）。"""

    feature_map: Dict[str, List[str]] = {}
    for col in df.columns:
        if not col.startswith("scaled_"):
            continue
        raw = col[len("scaled_"):]
        for suf in TIME_SUFFIXES:
            if raw.endswith(suf):
                base = raw[: -len(suf)]
                feature_map.setdefault(base, []).append(col)
                break
    # 确保列顺序为 T1 → T2 → T3（若存在）
    ordered_groups: List[List[str]] = []
    for base, cols in sorted(feature_map.items()):
        cols_sorted = []
        for suf in TIME_SUFFIXES:
            candidate = f"scaled_{base}{suf}"
            if candidate in cols:
                cols_sorted.append(candidate)
        ordered_groups.append(cols_sorted)
    return ordered_groups


def load_data() -> Tuple[pd.DataFrame, List[List[str]]]:
    df = read_smart(PROCESSED_DATA_PATH)
    feature_groups = collect_feature_sets(df)
    return df, feature_groups


def load_indices() -> Dict[int, Dict[str, np.ndarray]]:
    folds: Dict[int, Dict[str, np.ndarray]] = {}
    for k in range(1, 6):
        fold_dir = os.path.join(FOLDS_DIR, f"fold_{k}")
        train_idx = read_smart(os.path.join(fold_dir, "train.csv"))["idx"].to_numpy()
        val_idx = read_smart(os.path.join(fold_dir, "val.csv"))["idx"].to_numpy()
        test_idx = read_smart(os.path.join(fold_dir, "test.csv"))["idx"].to_numpy()
        folds[k] = {"train": train_idx, "val": val_idx, "test": test_idx}
    return folds


def mask_future_timepoints(df_split: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """在推理阶段将 __T2/__T3 列置为 NaN，保持列形状一致。"""

    masked = df_split.copy()
    for col in columns:
        for suf in TIME_SUFFIXES:
            if suf != "__T1" and col.endswith(suf):
                masked[col] = np.nan
                break
    return masked


def run_single_feature_models():
    df, feature_groups = load_data()
    folds = load_indices()

    tasks = [
        {"name": "T2", "label": "T2outcome"},
        {"name": "T3", "label": "T3outcome"},
    ]

    records = []

    for cols in tqdm(feature_groups, desc="单特征遍历"):
        feature_primary = cols[0]
        feature_name = feature_primary[len("scaled_") :].rsplit("__", 1)[0]
        for fold_id in range(1, 6):
            split_idx = folds[fold_id]
            train_idx, val_idx, test_idx = split_idx["train"], split_idx["val"], split_idx["test"]

            train_df = df.loc[train_idx, cols]
            val_df = mask_future_timepoints(df.loc[val_idx, cols], cols)
            test_df = mask_future_timepoints(df.loc[test_idx, cols], cols)

            train_matrix = train_df.to_numpy()
            val_matrix = val_df.to_numpy()
            test_matrix = test_df.to_numpy()

            # 以训练集均值填补，保证推理阶段不含 NaN
            train_means = np.nanmean(train_matrix, axis=0)
            train_matrix = np.nan_to_num(train_matrix, nan=train_means)
            val_matrix = np.nan_to_num(val_matrix, nan=train_means)
            test_matrix = np.nan_to_num(test_matrix, nan=train_means)

            for task in tasks:
                y_train = df.loc[train_idx, task["label"]].to_numpy()
                y_val = df.loc[val_idx, task["label"]].to_numpy()
                y_test = df.loc[test_idx, task["label"]].to_numpy()

                model = LogisticRegression(solver="liblinear", max_iter=500)
                try:
                    model.fit(train_matrix, y_train)
                except Exception:
                    # skip if training fails (e.g., single-class)
                    continue

                prob_val = model.predict_proba(val_matrix)[:, 1]
                prob_test = model.predict_proba(test_matrix)[:, 1]
                threshold = find_best_threshold_youden(y_val, prob_val)

                metrics_val = compute_metrics(y_val, prob_val, threshold)
                metrics_test = compute_metrics(y_test, prob_test, threshold)

                for split_name, split_metrics, probs, y_true, idx_arr in [
                    ("val", metrics_val, prob_val, y_val, val_idx),
                    ("test", metrics_test, prob_test, y_test, test_idx),
                ]:
                    rec = {
                        "feature": feature_primary,
                        "feature_base": feature_name,
                        "task": task["name"],
                        "fold": fold_id,
                        "split": split_name,
                        "threshold": threshold,
                    }
                    rec.update(split_metrics)
                    records.append(rec)

                # 保存预测详情便于后续绘图
                pred_df = pd.DataFrame(
                    {
                        "idx": np.concatenate([val_idx, test_idx]),
                        "split": ["val"] * len(val_idx) + ["test"] * len(test_idx),
                        "probability": np.concatenate([prob_val, prob_test]),
                        "label": np.concatenate([y_val, y_test]),
                        "feature": feature_primary,
                        "feature_base": feature_name,
                        "task": task["name"],
                        "fold": fold_id,
                    }
                )
                pred_out = os.path.join(
                    OUTPUT_DIR,
                    f"predictions_fold{fold_id}_{task['name']}_{feature_primary.replace('/', '_')}.csv",
                )
                pred_df.to_csv(pred_out, index=False, encoding="utf-8-sig")

    if records:
        metrics_df = pd.DataFrame(records)
        metrics_path = os.path.join(OUTPUT_DIR, "single_feature_lr_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
        print(f"[Done] 单特征 LR 指标已保存到: {metrics_path}")
    else:
        print("[Warning] 未生成任何指标，可能是训练数据存在问题。")


if __name__ == "__main__":
    run_single_feature_models()
