"""外部独立验证脚本：使用训练好的 spline_mixed + gradient_boosted 模型。

- 读取外部 6000 例仿真数据（包含 age/gender 列），对齐训练时的特征顺序。
- 训练阶段使用 T1/T2/T3，推理阶段自动将 __T2/__T3 掩码为 NaN（仅看 T1）。
- 加载第二次训练生成的 5 折模型，逐折推理并可选计算外部指标，然后输出折平均预测。
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = "/content/drive/MyDrive/Github项目/HCC预测机器学习"
EXTERNAL_CSV = "/content/drive/MyDrive/附三实验室资料/最终数据_仿真6000例_GaussianCopula_含年龄性别.csv"
BASE_FEATURES_PATH = os.path.join(PROJECT_ROOT, "base_feature_list.txt")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "第二次训练纵向推理模型")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "外部验证")
MODEL_KEY = "spline_mixed__gradient_boosted"
FOLDS = [1, 2, 3, 4, 5]

TASKS = (
    {"name": "T2", "label": "T2outcome", "time": "T2time"},
    {"name": "T3", "label": "T3outcome", "time": "T3time"},
)


def read_smart(path: str) -> pd.DataFrame:
    for enc in ("utf-8-sig", "gb18030", "gbk", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception:
            continue
    return pd.read_csv(path, low_memory=False)


def map_gender(value: object) -> float:
    s = str(value).strip().lower()
    if s in {"1", "m", "male", "男", "man", "male(男)"}:
        return 1.0
    if s in {"2", "f", "female", "女", "woman", "female(女)"}:
        return 0.0
    try:
        v = int(float(s))
        if v == 1:
            return 1.0
        if v in (0, 2):
            return 0.0
    except Exception:
        pass
    return np.nan


def load_base_features() -> List[str]:
    with open(BASE_FEATURES_PATH, "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def fill_feature_missing(feature_df: pd.DataFrame) -> pd.DataFrame:
    missing_total = int(feature_df.isna().sum().sum())
    if missing_total == 0:
        return feature_df
    col_means = feature_df.mean(skipna=True).fillna(0.0)
    print(f"[Info] 检测到 {missing_total} 个缺失值，已使用对应列的均值进行填补。")
    return feature_df.fillna(col_means)


def expand_static_timepoints(feature_df: pd.DataFrame) -> pd.DataFrame:
    for base in ("age", "sex"):
        t1_col = f"{base}__T1"
        if t1_col not in feature_df.columns:
            continue
        for suffix in ("__T2", "__T3"):
            col = f"{base}{suffix}"
            if col not in feature_df.columns:
                feature_df[col] = feature_df[t1_col]
    return feature_df


def enforce_available_history(feature_df: pd.DataFrame) -> pd.DataFrame:
    df = feature_df.copy()
    for suffix in ("__T2", "__T3"):
        cols = [c for c in df.columns if c.endswith(suffix)]
        for col in cols:
            df[col] = np.nan
    return df


def build_feature_frame() -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    df = read_smart(EXTERNAL_CSV)
    base_features = load_base_features()

    df = df.rename(columns={"subject_id": "subject_id_main"}, errors="ignore")
    if "subject_id_main" not in df.columns:
        df = df.rename(columns={df.columns[0]: "subject_id_main"}, errors="ignore")
    df["subject_id_main"] = df["subject_id_main"].astype(str)

    # 年龄/性别：外部队列已提供 age/gender 列
    if "age__T1" not in df.columns:
        if "age" not in df.columns:
            raise RuntimeError("外部数据缺少 age 列")
        df["age__T1"] = pd.to_numeric(df["age"], errors="coerce")
    if "sex__T1" not in df.columns:
        gender_col = "gender" if "gender" in df.columns else None
        if gender_col is None:
            raise RuntimeError("外部数据缺少 gender/sex 列")
        df["sex__T1"] = df[gender_col].apply(map_gender)

    feature_columns: List[str] = ["age__T1", "sex__T1"]
    for base in base_features:
        cols = [f"{base}__T1", f"{base}__T2", f"{base}__T3"]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise RuntimeError(f"外部数据缺少 {base} 的列: {missing}")
        feature_columns.extend(cols)

    feature_df = df[feature_columns].copy()
    feature_df = fill_feature_missing(feature_df)
    feature_df = expand_static_timepoints(feature_df)
    meta_cols = {"subject_id", "T2outcome", "T3outcome", "T2time", "T3time"}
    meta_present = [c for c in meta_cols if c in df.columns]
    meta_df = df[[c for c in meta_present]].copy()
    if "subject_id" not in meta_df.columns:
        meta_df["subject_id"] = df["subject_id_main"]
    return feature_df, meta_df, base_features


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict:
    mask = ~np.isnan(y_true) & ~np.isnan(y_prob)
    y_true = y_true[mask].astype(int)
    y_prob = y_prob[mask]
    if len(y_true) == 0:
        return {}
    auc = pr_auc = float("nan")
    if len(np.unique(y_true)) > 1:
        auc = float(roc_auc_score(y_true, y_prob))
        pr_auc = float(average_precision_score(y_true, y_prob))
    y_pred = (y_prob >= threshold).astype(int)
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    except ValueError:
        tn = fp = fn = tp = 0
        if y_true.sum() == 0:
            tn = len(y_true)
        else:
            tp = len(y_true)
    sensitivity = float(recall_score(y_true, y_pred, zero_division=0))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    ppv = float(precision_score(y_true, y_pred, zero_division=0))
    npv = float(tn / (tn + fn)) if (tn + fn) > 0 else float("nan")
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else float("nan")
    brier = float(np.mean((y_true - y_prob) ** 2))
    return {
        "AUC": auc,
        "PR_AUC": pr_auc,
        "Accuracy": acc,
        "Balanced_Accuracy": bal_acc,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "PPV": ppv,
        "NPV": npv,
        "F1": f1,
        "MCC": mcc,
        "Brier": brier,
        "Confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def load_model_bundle(fold: int) -> Tuple[Dict, Dict[str, float]]:
    run_dir = os.path.join(RESULTS_DIR, f"fold_{fold}", MODEL_KEY)
    bundle = load(os.path.join(run_dir, "model.joblib"))
    metrics_path = os.path.join(run_dir, "metrics.json")
    with open(metrics_path, "r", encoding="utf-8") as fh:
        metrics = json.load(fh)
    thresholds = {task["name"]: metrics["tasks"][task["name"]]["threshold"] for task in TASKS}
    return bundle, thresholds


def predict_fold(
    fold: int,
    bundle: Dict,
    thresholds: Dict[str, float],
    feature_df: pd.DataFrame,
    meta_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    encoder = bundle["encoder"]
    risk_model = bundle["risk_model"]
    fill_values = bundle["embedding_fill_values"]

    masked_df = enforce_available_history(feature_df)
    X = encoder.transform(masked_df)
    X = np.asarray(X, dtype=float)
    X[~np.isfinite(X)] = np.nan
    # 如果存在 NaN，用训练期的填充值替换
    if np.isnan(X).any():
        rows, cols = np.where(~np.isfinite(X))
        X = X.copy()
        X[rows, cols] = fill_values[cols]

    prob_dict = risk_model.predict_proba(X)
    preds: List[pd.DataFrame] = []
    metrics: Dict[str, Dict] = {}
    for idx, task in enumerate(TASKS):
        probs = prob_dict[f"task_{idx}"]
        df_rows = {
            "subject_id": meta_df.get("subject_id", pd.Series(np.arange(len(probs)))).to_numpy(),
            "fold": fold,
            "task": task["name"],
            "probability": probs,
        }
        if task["label"] in meta_df.columns:
            labels = meta_df[task["label"]].to_numpy()
            df_rows["label"] = labels
            thr = thresholds.get(task["name"], 0.5)
            metrics[task["name"]] = compute_metrics(labels, probs, thr)
            df_rows["pred_label"] = (probs >= thr).astype(int)
            df_rows["threshold"] = thr
        preds.append(pd.DataFrame(df_rows))
    return pd.concat(preds, ignore_index=True), metrics


def aggregate_predictions(all_preds: pd.DataFrame) -> pd.DataFrame:
    grouped = all_preds.groupby(["subject_id", "task"], as_index=False)["probability"].mean()
    grouped = grouped.rename(columns={"probability": "prob_mean"})
    return grouped


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    feature_df, meta_df, base_features = build_feature_frame()

    all_fold_preds: List[pd.DataFrame] = []
    metrics_rows: List[Dict] = []
    for fold in FOLDS:
        bundle, thresholds = load_model_bundle(fold)
        preds, mt = predict_fold(fold, bundle, thresholds, feature_df, meta_df)
        all_fold_preds.append(preds)
        preds.to_csv(
            os.path.join(OUTPUT_DIR, f"predictions_fold{fold}.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        for task_name, payload in mt.items():
            row = {"fold": fold, "task": task_name}
            row.update(payload)
            metrics_rows.append(row)
    all_preds = pd.concat(all_fold_preds, ignore_index=True)
    all_preds.to_csv(os.path.join(OUTPUT_DIR, "predictions_all_folds.csv"), index=False, encoding="utf-8-sig")

    ensemble = aggregate_predictions(all_preds)
    ensemble.to_csv(
        os.path.join(OUTPUT_DIR, "predictions_ensemble.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    if metrics_rows:
        pd.DataFrame(metrics_rows).to_csv(
            os.path.join(OUTPUT_DIR, "metrics_external.csv"), index=False, encoding="utf-8-sig"
        )
    print("[Done] 外部验证推理完成，结果已保存到", OUTPUT_DIR)


if __name__ == "__main__":
    main()
