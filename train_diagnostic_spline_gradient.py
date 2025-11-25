"""诊断任务：SplineMixedEffectEncoder + HistGradientBoostingClassifier.

* 训练阶段：使用全部 T1/T2/T3 特征学习轨迹（与最佳模型一致）。
* 推理阶段：仅保留少数可用指标，其余列全部掩码为 NaN，确保诊断时只依赖有限化验项。
* 输出：每折预测、指标以及阈值文件写入 RESULT_DIR（默认“诊断模型”子目录）。

使用方式（Colab/本地均可）：
  python train_diagnostic_spline_gradient.py
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_array, check_is_fitted
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 路径与常量
# ---------------------------------------------------------------------------
PROJECT_ROOT = "/content/drive/MyDrive/Github项目/HCC预测机器学习"
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "processed_ML_data.csv")
SCALER_PATH = os.path.join(PROJECT_ROOT, "feature_scaler.joblib")
FOLDS_DIR = os.path.join(PROJECT_ROOT, "folds")
RESULT_DIR = os.path.join(PROJECT_ROOT, "诊断模型")
os.makedirs(RESULT_DIR, exist_ok=True)

TIME_SUFFIXES = ("__T1", "__T2", "__T3")
TASK = {"name": "Diagnosis_T3", "label_col": "T3outcome"}
# 推理可用的少量基准指标（可按需调整）
AVAILABLE_BASE_FEATURES = [
    "甲胎蛋白(AFP)",
    "高尔基体蛋白73(GP73)",
    "异常凝血酶原(PIVKA-II)",
    "甲胎蛋白异质体(AFP-L3%)",
    "凝血酶原标准化比值(PT-INR)",
    "活化部分凝血活酶比值(APTT-ratio)",
    "谷氨酰转肽酶(GGT)",
    "血红蛋白浓度(HGB)",
    "血小板计数(PLT)",
]

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def read_smart(path: str) -> pd.DataFrame:
    for enc in ["utf-8-sig", "gb18030", "gbk", "utf-8"]:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception:
            pass
    return pd.read_csv(path, low_memory=False)


def load_fold_indices() -> Dict[int, Dict[str, np.ndarray]]:
    folds: Dict[int, Dict[str, np.ndarray]] = {}
    for k in range(1, 6):
        fold_dir = os.path.join(FOLDS_DIR, f"fold_{k}")
        splits = {}
        for split in ("train", "val", "test"):
            csv_path = os.path.join(fold_dir, f"{split}.csv")
            splits[split] = read_smart(csv_path)["idx"].to_numpy(dtype=int)
        folds[k] = splits
    return folds


# ---------------------------------------------------------------------------
# 轨迹编码器与风险模型（内联，避免反序列化依赖）
# ---------------------------------------------------------------------------

def _ensure_times(base_name: str, df: pd.DataFrame) -> List[str]:
    cols = [f"{base_name}{s}" for s in TIME_SUFFIXES]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"缺少 {base_name} 的列: {missing}")
    return cols


def _time_design_matrix(times: Sequence[float], degree: int) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    return np.vstack([times ** k for k in range(degree + 1)]).T


def _solve_coeffs_safe(design: np.ndarray, values: np.ndarray) -> np.ndarray:
    mask = np.isfinite(values)
    if not mask.any():
        return np.zeros(design.shape[1], dtype=float)
    design_sub = design[mask]
    values_sub = values[mask]
    coeffs, _, _, _ = np.linalg.lstsq(design_sub, values_sub, rcond=None)
    if len(coeffs) < design.shape[1]:
        coeffs = np.pad(coeffs, (0, design.shape[1] - len(coeffs)), constant_values=0.0)
    return coeffs


class SplineMixedEffectEncoder(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        base_features: Iterable[str],
        degree: int = 2,
        time_points: Tuple[float, float, float] = (0.0, 1.0, 2.0),
    ) -> None:
        self.base_features = list(base_features)
        self.degree = degree
        self.time_points = tuple(time_points)

    def fit(self, df: pd.DataFrame, y=None):
        self.base_features_ = list(self.base_features)
        self.design_ = _time_design_matrix(self.time_points, self.degree)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "design_")
        embeddings: List[np.ndarray] = []
        for base in self.base_features_:
            cols = _ensure_times(base, df)
            values = df[cols].to_numpy(dtype=float)
            coeffs = [_solve_coeffs_safe(self.design_, row) for row in values]
            embeddings.append(np.vstack(coeffs))
        return np.hstack(embeddings)

    def get_feature_names_out(self) -> List[str]:
        check_is_fitted(self, "design_")
        names: List[str] = []
        for base in self.base_features_:
            for k in range(self.degree + 1):
                names.append(f"{base}__coef_{k}")
        return names


class GradientBoostedRiskModel(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth: int = 3, learning_rate: float = 0.1, max_iter: int = 200):
        self.model = HistGradientBoostingClassifier(
            max_depth=max_depth,
            learning_rate=learning_rate,
            max_iter=max_iter,
            random_state=42,
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = check_array(X)
        y = np.asarray(y)
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self.model)
        return self.model.predict_proba(check_array(X))


# ---------------------------------------------------------------------------
# 特征准备与掩码
# ---------------------------------------------------------------------------

def load_feature_frame() -> Tuple[pd.DataFrame, np.ndarray]:
    df = read_smart(PROCESSED_DATA_PATH)
    scaler = load(SCALER_PATH)
    scaled_cols = [c for c in df.columns if c.startswith("scaled_")]
    feature_names = [c.replace("scaled_", "", 1) for c in scaled_cols]
    X_scaled = df[scaled_cols].to_numpy()
    X_raw = scaler.inverse_transform(X_scaled)
    feature_df = pd.DataFrame(X_raw, columns=feature_names)
    # 填补缺失
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    for col in feature_df.columns:
        if feature_df[col].isna().any():
            feature_df[col] = feature_df[col].fillna(feature_df[col].mean())
    labels = df[TASK["label_col"]].to_numpy()
    return feature_df, labels


def expand_static_timepoints(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for static_col in ("age__T1", "sex__T1"):
        if static_col in df.columns:
            base = static_col.replace("__T1", "")
            for suffix in ("__T2", "__T3"):
                new_col = f"{base}{suffix}"
                if new_col not in df.columns:
                    df[new_col] = df[static_col]
    return df


def mask_for_inference(df: pd.DataFrame, allowed_bases: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    allowed = set(allowed_bases)
    for col in df.columns:
        base = col.split("__T")[0]
        if base not in allowed:
            df[col] = np.nan
    return df


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------

def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    mask = ~np.isnan(y_true) & ~np.isnan(y_prob)
    y_true = y_true[mask]
    y_prob = y_prob[mask]
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    idx = np.nanargmax(tpr - fpr)
    return float(np.clip(thresholds[idx], 0.0, 1.0))


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    mask = ~np.isnan(y_true) & ~np.isnan(y_prob)
    y_true = y_true[mask]
    y_prob = y_prob[mask]
    y_pred = (y_prob >= threshold).astype(int)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    try:
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = float(np.trapz(rec, prec))
    except Exception:
        pr_auc = float("nan")
    return {
        "AUC": auc,
        "PR_AUC": pr_auc,
        "Sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": recall_score(1 - y_true, 1 - y_pred, zero_division=0),
        "PPV": precision_score(y_true, y_pred, zero_division=0),
        "NPV": precision_score(1 - y_true, 1 - y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    feature_df, labels = load_feature_frame()
    feature_df = expand_static_timepoints(feature_df)

    # 轨迹编码基准列表（用于训练，包含所有 base feature）
    base_features = sorted({col.split("__T")[0] for col in feature_df.columns if col.endswith(TIME_SUFFIXES)})

    folds = load_fold_indices()
    results: List[Dict] = []

    for fold_id in range(1, 6):
        splits = folds[fold_id]
        train_df = feature_df.iloc[splits["train"]].reset_index(drop=True)
        val_df = feature_df.iloc[splits["val"]].reset_index(drop=True)
        test_df = feature_df.iloc[splits["test"]].reset_index(drop=True)

        train_y = labels[splits["train"]]
        val_y = labels[splits["val"]]
        test_y = labels[splits["test"]]

        encoder = SplineMixedEffectEncoder(base_features=base_features, degree=2)
        encoder.fit(train_df)
        X_train = encoder.transform(train_df)

        risk = GradientBoostedRiskModel(max_depth=3, learning_rate=0.1, max_iter=300)
        risk.fit(X_train, train_y)

        # 推理阶段仅保留少量可用指标
        val_masked = mask_for_inference(val_df, AVAILABLE_BASE_FEATURES)
        test_masked = mask_for_inference(test_df, AVAILABLE_BASE_FEATURES)

        X_val = encoder.transform(val_masked)
        X_test = encoder.transform(test_masked)

        val_prob = risk.predict_proba(X_val)[:, 1]
        test_prob = risk.predict_proba(X_test)[:, 1]

        thr = find_best_threshold(val_y, val_prob)
        metrics_val = compute_metrics(val_y, val_prob, thr)
        metrics_test = compute_metrics(test_y, test_prob, thr)

        fold_dir = os.path.join(RESULT_DIR, f"fold_{fold_id}")
        os.makedirs(fold_dir, exist_ok=True)

        dump({"encoder": encoder, "risk": risk, "threshold": thr}, os.path.join(fold_dir, "model.joblib"))
        pd.DataFrame({"idx": splits["val"], "prob": val_prob, "label": val_y}).to_csv(
            os.path.join(fold_dir, "pred_val.csv"), index=False, encoding="utf-8-sig"
        )
        pd.DataFrame({"idx": splits["test"], "prob": test_prob, "label": test_y}).to_csv(
            os.path.join(fold_dir, "pred_test.csv"), index=False, encoding="utf-8-sig"
        )

        metrics_path = os.path.join(fold_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as fh:
            json.dump({"threshold": thr, "val": metrics_val, "test": metrics_test}, fh, ensure_ascii=False, indent=2)

        results.append({
            "fold": fold_id,
            "threshold": thr,
            **{f"val_{k}": v for k, v in metrics_val.items()},
            **{f"test_{k}": v for k, v in metrics_test.items()},
        })
        print(f"[Done] Fold {fold_id} 训练与推理完成")

    pd.DataFrame(results).to_csv(os.path.join(RESULT_DIR, "summary.csv"), index=False, encoding="utf-8-sig")
    print(f"\n[Done] 全部折次完成，汇总已保存到 {RESULT_DIR}")


if __name__ == "__main__":
    main()
