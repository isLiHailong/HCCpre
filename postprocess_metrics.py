"""重算纵向模型的指标并汇总输出表。"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
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
RESULTS_DIR = os.path.join(PROJECT_ROOT, "第二次训练纵向推理模型")
TASKS = (
    {"name": "T2", "label": "T2outcome", "time": "T2time"},
    {"name": "T3", "label": "T3outcome", "time": "T3time"},
)
SPLITS = ("train", "val", "test")
AGGREGATE_PATH = os.path.join(RESULTS_DIR, "metrics_aggregate_new.csv")


class NpEncoder(json.JSONEncoder):
    def default(self, obj):  # type: ignore[override]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super().default(obj)


def read_smart(path: str) -> pd.DataFrame:
    for enc in ("utf-8-sig", "gb18030", "gbk", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception:
            continue
    return pd.read_csv(path, low_memory=False)


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    from sklearn.metrics import roc_curve

    mask = ~np.isnan(y_true) & ~np.isnan(y_prob)
    y_true = y_true[mask]
    y_prob = y_prob[mask]
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden = tpr - fpr
    idx = np.nanargmax(youden)
    thr = thresholds[idx]
    return float(np.clip(thr, 0.0, 1.0))


def concordance_index(times: np.ndarray, events: np.ndarray, scores: np.ndarray) -> float:
    mask = (~np.isnan(events)) & (~np.isnan(scores))
    events = events[mask]
    scores = scores[mask]
    if len(np.unique(events)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(events, scores))
    except ValueError:
        return float("nan")


def build_time_curves(y_true: np.ndarray, y_prob: np.ndarray, times: np.ndarray) -> Tuple[List[Dict], List[Dict]]:
    mask = ~np.isnan(times)
    if mask.sum() < 2:
        return [], []
    y_true = y_true[mask]
    y_prob = y_prob[mask]
    times = times[mask]
    order = np.argsort(times)
    times = times[order]
    y_true = y_true[order]
    y_prob = y_prob[order]
    unique_times = np.unique(times)
    auc_curve: List[Dict] = []
    brier_curve: List[Dict] = []
    for tau in unique_times:
        subset = times <= tau
        if subset.sum() < 10:
            continue
        y_tau = y_true[subset]
        p_tau = y_prob[subset]
        auc_val = float("nan")
        if len(np.unique(y_tau)) > 1:
            try:
                auc_val = float(roc_auc_score(y_tau, p_tau))
            except ValueError:
                auc_val = float("nan")
        auc_curve.append({"time": float(tau), "auc": auc_val, "n_samples": int(subset.sum())})
        brier_curve.append(
            {
                "time": float(tau),
                "brier": float(np.mean((y_tau - p_tau) ** 2)),
                "n_samples": int(subset.sum()),
            }
        )
    return auc_curve, brier_curve


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, times: np.ndarray) -> Dict:
    mask = ~np.isnan(y_true) & ~np.isnan(y_prob)
    y_true = y_true[mask].astype(int)
    y_prob = y_prob[mask]
    times = times[mask]
    if len(y_true) == 0:
        return {}
    auc = float("nan")
    pr_auc = float("nan")
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
    c_index = concordance_index(times, y_true, y_prob)
    auc_curve, brier_curve = build_time_curves(y_true, y_prob, times)
    ibs = float(np.nanmean([entry["brier"] for entry in brier_curve])) if brier_curve else float("nan")
    cal_y, cal_x = [], []
    if len(np.unique(y_true)) > 1 and len(y_true) >= 20:
        cal_y, cal_x = calibration_curve(y_true, y_prob, n_bins=10)
    return {
        "AUC": auc,
        "PR_AUC": pr_auc,
        "C_index": float(c_index),
        "Accuracy": acc,
        "Balanced_Accuracy": bal_acc,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "PPV": ppv,
        "NPV": npv,
        "F1": f1,
        "MCC": mcc,
        "Brier": brier,
        "Integrated_Brier": ibs,
        "Confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "Calibration_Curve": {
            "mean_pred": cal_x.tolist() if len(cal_x) else [],
            "frac_pos": cal_y.tolist() if len(cal_y) else [],
        },
        "Time_Dependent_AUC": auc_curve,
        "Time_Dependent_Brier": brier_curve,
    }


def recompute_for_run(pred_path: str, fold: int, trajectory: str, risk: str) -> Dict:
    df = read_smart(pred_path)
    payload = {
        "fold": fold,
        "trajectory": trajectory,
        "risk": risk,
        "generated_at": datetime.utcnow().isoformat(),
        "tasks": {},
        "source_predictions": os.path.relpath(pred_path, PROJECT_ROOT),
    }
    for task in TASKS:
        task_df = df[df["task"] == task["name"]]
        if task_df.empty:
            continue
        split_frames = {split: task_df[task_df["split"] == split] for split in SPLITS}
        val_frame = split_frames["val"]
        if val_frame.empty:
            threshold = 0.5
        else:
            threshold = find_best_threshold(
                val_frame["label"].to_numpy(dtype=float),
                val_frame["probability"].to_numpy(dtype=float),
            )
        payload["tasks"][task["name"]] = {"threshold": threshold, "splits": {}}
        for split_name, frame in split_frames.items():
            if frame.empty:
                payload["tasks"][task["name"]]["splits"][split_name] = {}
                continue
            metrics = compute_metrics(
                frame["label"].to_numpy(dtype=float),
                frame["probability"].to_numpy(dtype=float),
                threshold,
                frame["time"].to_numpy(dtype=float),
            )
            payload["tasks"][task["name"]]["splits"][split_name] = metrics
    return payload


def aggregate_rows(payload: Dict) -> List[Dict]:
    rows: List[Dict] = []
    scalar_fields = [
        "AUC",
        "PR_AUC",
        "C_index",
        "Accuracy",
        "Balanced_Accuracy",
        "Sensitivity",
        "Specificity",
        "PPV",
        "NPV",
        "F1",
        "MCC",
        "Brier",
        "Integrated_Brier",
    ]
    for task_name, info in payload.get("tasks", {}).items():
        threshold = info.get("threshold")
        for split_name, metrics in info.get("splits", {}).items():
            if not metrics:
                continue
            row = {
                "fold": payload["fold"],
                "trajectory": payload["trajectory"],
                "risk": payload["risk"],
                "task": task_name,
                "split": split_name,
                "threshold": threshold,
            }
            for field in scalar_fields:
                row[field] = metrics.get(field)
            confusion = metrics.get("Confusion", {})
            row.update(
                {
                    "TN": confusion.get("tn"),
                    "FP": confusion.get("fp"),
                    "FN": confusion.get("fn"),
                    "TP": confusion.get("tp"),
                }
            )
            rows.append(row)
    return rows


def main() -> None:
    if not os.path.isdir(RESULTS_DIR):
        raise SystemExit(f"结果目录不存在: {RESULTS_DIR}")
    aggregate: List[Dict] = []
    processed = 0
    for fold_name in sorted(os.listdir(RESULTS_DIR)):
        if not fold_name.startswith("fold_"):
            continue
        fold_dir = os.path.join(RESULTS_DIR, fold_name)
        if not os.path.isdir(fold_dir):
            continue
        fold_idx = int(fold_name.split("_")[1])
        for combo in sorted(os.listdir(fold_dir)):
            combo_dir = os.path.join(fold_dir, combo)
            if not os.path.isdir(combo_dir):
                continue
            pred_path = os.path.join(combo_dir, "predictions.csv")
            if not os.path.exists(pred_path):
                continue
            trajectory, risk = combo.split("__", 1)
            payload = recompute_for_run(pred_path, fold_idx, trajectory, risk)
            metrics_path = os.path.join(combo_dir, "metrics_new.json")
            with open(metrics_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2, cls=NpEncoder)
            aggregate.extend(aggregate_rows(payload))
            processed += 1
            print(f"[Done] {fold_name}/{combo} -> metrics_new.json")
    if aggregate:
        df = pd.DataFrame(aggregate)
        df.to_csv(AGGREGATE_PATH, index=False, encoding="utf-8-sig")
        print(f"汇总表已保存到: {AGGREGATE_PATH}")
    else:
        print("未找到可处理的 predictions.csv 文件。")
    print(f"共处理 {processed} 个模型目录。")


if __name__ == "__main__":
    main()
