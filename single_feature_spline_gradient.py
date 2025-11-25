import os
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import load
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
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
from sklearn.utils.validation import check_array, check_is_fitted
from tqdm import tqdm

warnings.filterwarnings("ignore")

PROJECT_ROOT = "/content/drive/MyDrive/Github项目/HCC预测机器学习"
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "processed_ML_data.csv")
SCALER_PATH = os.path.join(PROJECT_ROOT, "feature_scaler.joblib")
FOLDS_DIR = os.path.join(PROJECT_ROOT, "folds")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "单特征spline_mixed_gradient")

os.makedirs(OUTPUT_DIR, exist_ok=True)

TIME_SUFFIXES = ("__T1", "__T2", "__T3")
INFERENCE_ALLOWED_SUFFIXES = ("__T1",)


# ---------------------------------------------------------------------------
# Minimal inline implementations of the trajectory encoder and risk model
# ---------------------------------------------------------------------------


def _ensure_times(base_name: str, df: pd.DataFrame) -> List[str]:
    suffixes = ["__T1", "__T2", "__T3"]
    cols = [f"{base_name}{suffix}" for suffix in suffixes]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"缺少 {base_name} 的列: {missing}")
    return cols


def _time_design_matrix(times: Tuple[float, ...], degree: int) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    cols = [times ** k for k in range(degree + 1)]
    return np.vstack(cols).T


def _solve_coeffs_safe(design: np.ndarray, values: np.ndarray) -> np.ndarray:
    """
    Solve polynomial coefficients for a single sample, tolerating missing time points.

    * If部分时间点缺失（NaN/inf），只用可用时间点的 design 子矩阵进行最小二乘；
    * 如果所有时间点都缺失，返回零向量，由后续的 fill 逻辑再做补齐。
    """

    mask = np.isfinite(values)
    if not mask.any():
        return np.zeros(design.shape[1], dtype=float)

    design_sub = design[mask]
    values_sub = values[mask]

    # 处理只有 1 个观测点时的秩亏情况：np.linalg.lstsq 仍可返回最小二乘解
    coeffs, _, _, _ = np.linalg.lstsq(design_sub, values_sub, rcond=None)
    # 对于极端情况下返回的长度不足 degree+1 的系数，做右侧零填充
    if len(coeffs) < design.shape[1]:
        coeffs = np.pad(coeffs, (0, design.shape[1] - len(coeffs)), constant_values=0.0)
    return coeffs


class SplineMixedEffectEncoder(TransformerMixin, BaseEstimator):
    """Use low-degree polynomial coefficients as trajectory embeddings."""

    def __init__(
        self,
        base_features: List[str],
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
        for base_name in self.base_features_:
            cols = _ensure_times(base_name, df)
            values = df[cols].to_numpy(dtype=float)

            # 针对每个样本单独计算系数，忽略缺失的时间点
            coeff_list = [
                _solve_coeffs_safe(self.design_, row)
                for row in values
            ]
            embeddings.append(np.vstack(coeff_list))

        return np.hstack(embeddings)


class GradientBoostedRiskModel(BaseEstimator, ClassifierMixin):
    """Per-task HistGradientBoosting classifier wrapper."""

    def __init__(self, max_depth: int = 3, learning_rate: float = 0.1, max_iter: int = 200):
        from sklearn.ensemble import HistGradientBoostingClassifier

        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self._hgb_cls = HistGradientBoostingClassifier

    def fit(self, X: np.ndarray, Y: np.ndarray):
        X = check_array(X)
        Y = np.asarray(Y)
        self.models_ = []
        for col in range(Y.shape[1]):
            clf = self._hgb_cls(
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                max_iter=self.max_iter,
                random_state=42,
            )
            clf.fit(X, Y[:, col])
            self.models_.append(clf)
        return self

    def predict_proba(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        check_is_fitted(self, "models_")
        X = check_array(X)
        return {f"task_{i}": model.predict_proba(X)[:, 1] for i, model in enumerate(self.models_)}


def read_smart(path: str) -> pd.DataFrame:
    for enc in ["utf-8-sig", "gb18030", "gbk", "utf-8"]:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception:
            continue
    return pd.read_csv(path, low_memory=False)


def fill_feature_missing(df: pd.DataFrame) -> pd.DataFrame:
    filled = df.copy()
    for col in filled.columns:
        vals = filled[col].to_numpy(dtype=float)
        mask = np.isfinite(vals)
        if not mask.any():
            filled[col] = 0.0
        else:
            mean_val = vals[mask].mean()
            vals[~mask] = mean_val
            filled[col] = vals
    return filled


def expand_static_timepoints(feature_df: pd.DataFrame) -> pd.DataFrame:
    static_bases = ("age", "sex")
    for base in static_bases:
        t1_col = f"{base}__T1"
        if t1_col not in feature_df.columns:
            continue
        for suffix in ("__T2", "__T3"):
            col = f"{base}{suffix}"
            if col not in feature_df.columns:
                feature_df[col] = feature_df[t1_col]
    return feature_df


def load_feature_frame() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = read_smart(PROCESSED_DATA_PATH)
    scaler = load(SCALER_PATH)
    scaled_cols = [c for c in df.columns if c.startswith("scaled_")]
    feature_names = [c.replace("scaled_", "", 1) for c in scaled_cols]
    X_scaled = df[scaled_cols].to_numpy()
    X_raw = scaler.inverse_transform(X_scaled)
    feature_df = pd.DataFrame(X_raw, columns=feature_names)
    feature_df = fill_feature_missing(feature_df)
    feature_df = expand_static_timepoints(feature_df)
    meta_df = df[["T2outcome", "T3outcome"]].copy()
    return feature_df, meta_df


def load_indices() -> Dict[int, Dict[str, np.ndarray]]:
    folds: Dict[int, Dict[str, np.ndarray]] = {}
    for k in range(1, 6):
        fold_dir = os.path.join(FOLDS_DIR, f"fold_{k}")
        folds[k] = {
            split: read_smart(os.path.join(fold_dir, f"{split}.csv"))["idx"].to_numpy(dtype=int)
            for split in ("train", "val", "test")
        }
    return folds


def collect_feature_sets(df: pd.DataFrame) -> List[Tuple[str, List[str]]]:
    feature_map: Dict[str, List[str]] = {}
    for col in df.columns:
        for suf in TIME_SUFFIXES:
            if col.endswith(suf):
                base = col[: -len(suf)]
                feature_map.setdefault(base, []).append(col)
                break
    ordered: List[Tuple[str, List[str]]] = []
    for base, cols in sorted(feature_map.items()):
        seq = []
        for suf in TIME_SUFFIXES:
            name = f"{base}{suf}"
            if name in cols:
                seq.append(name)
        ordered.append((base, seq))
    return ordered


def mask_future_timepoints(df_split: pd.DataFrame) -> pd.DataFrame:
    masked = df_split.copy()
    allowed = set(INFERENCE_ALLOWED_SUFFIXES)
    for suffix in TIME_SUFFIXES:
        if suffix in allowed:
            continue
        cols = [c for c in masked.columns if c.endswith(suffix)]
        for col in cols:
            masked[col] = np.nan
    return masked


def find_best_threshold_youden(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    mask = ~np.isnan(y_true) & ~np.isnan(y_prob)
    if mask.sum() < 2 or len(np.unique(y_true[mask])) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true[mask], y_prob[mask])
    youden = tpr - fpr
    idx = np.nanargmax(youden)
    th = thresholds[idx]
    return float(np.clip(th, 0.0, 1.0))


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    mask = ~np.isnan(y_true) & ~np.isnan(y_prob)
    y_true = y_true[mask]
    y_prob = y_prob[mask]
    if len(y_true) == 0:
        return {k: np.nan for k in [
            "AUC",
            "PR_AUC",
            "Accuracy",
            "Balanced_Accuracy",
            "Sensitivity",
            "Specificity",
            "PPV",
            "NPV",
            "F1",
            "MCC",
            "Brier",
        ]}
    metrics["AUC"] = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    metrics["PR_AUC"] = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["Accuracy"] = accuracy_score(y_true, y_pred)
    metrics["Balanced_Accuracy"] = balanced_accuracy_score(y_true, y_pred)
    metrics["Sensitivity"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["Specificity"] = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    metrics["PPV"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["NPV"] = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    metrics["F1"] = f1_score(y_true, y_pred, zero_division=0)
    metrics["MCC"] = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_pred)) > 1 else np.nan
    metrics["Brier"] = brier_score_loss(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    return metrics


def compute_embedding_fill_values(emb: np.ndarray) -> np.ndarray:
    fill = np.nanmean(np.where(np.isfinite(emb), emb, np.nan), axis=0)
    fill = np.where(np.isfinite(fill), fill, 0.0)
    return fill


def apply_embedding_fill(emb: np.ndarray, fill: np.ndarray) -> np.ndarray:
    emb = emb.copy()
    bad = ~np.isfinite(emb)
    if bad.any():
        emb[bad] = np.take(fill, np.nonzero(bad)[1])
    return emb


def run_single_feature_models():
    feature_df, meta_df = load_feature_frame()
    feature_groups = collect_feature_sets(feature_df)
    folds = load_indices()

    tasks = [
        {"name": "T2", "label": "T2outcome"},
        {"name": "T3", "label": "T3outcome"},
    ]

    records: List[Dict[str, float]] = []

    for base, cols in tqdm(feature_groups, desc="单特征遍历"):
        for fold_id in range(1, 6):
            idxs = folds[fold_id]
            train_idx, val_idx, test_idx = idxs["train"], idxs["val"], idxs["test"]

            train_df = feature_df.loc[train_idx, cols]
            val_df = mask_future_timepoints(feature_df.loc[val_idx, cols])
            test_df = mask_future_timepoints(feature_df.loc[test_idx, cols])

            encoder = SplineMixedEffectEncoder(base_features=[base])
            encoder.fit(train_df)

            train_emb = encoder.transform(train_df)
            fill_values = compute_embedding_fill_values(train_emb)
            train_emb = apply_embedding_fill(train_emb, fill_values)

            Y_train = np.column_stack(
                [meta_df.loc[train_idx, task["label"]].to_numpy() for task in tasks]
            )

            risk_model = GradientBoostedRiskModel()
            try:
                risk_model.fit(train_emb, Y_train)
            except Exception:
                continue

            split_embs: Dict[str, np.ndarray] = {}
            for split_name, df_split in (("val", val_df), ("test", test_df)):
                emb = encoder.transform(df_split)
                emb = apply_embedding_fill(emb, fill_values)
                split_embs[split_name] = emb

            prob_val = risk_model.predict_proba(split_embs["val"])
            prob_test = risk_model.predict_proba(split_embs["test"])

            for task_idx, task in enumerate(tasks):
                y_val = meta_df.loc[val_idx, task["label"]].to_numpy()
                y_test = meta_df.loc[test_idx, task["label"]].to_numpy()

                val_probs = prob_val[f"task_{task_idx}"]
                test_probs = prob_test[f"task_{task_idx}"]
                threshold = find_best_threshold_youden(y_val, val_probs)

                metrics_val = compute_metrics(y_val, val_probs, threshold)
                metrics_test = compute_metrics(y_test, test_probs, threshold)

                for split_name, split_metrics, probs, y_true, idx_arr in [
                    ("val", metrics_val, val_probs, y_val, val_idx),
                    ("test", metrics_test, test_probs, y_test, test_idx),
                ]:
                    rec = {
                        "feature_base": base,
                        "task": task["name"],
                        "fold": fold_id,
                        "split": split_name,
                        "threshold": threshold,
                    }
                    rec.update(split_metrics)
                    records.append(rec)

                pred_df = pd.DataFrame(
                    {
                        "idx": np.concatenate([val_idx, test_idx]),
                        "split": ["val"] * len(val_idx) + ["test"] * len(test_idx),
                        "probability": np.concatenate([val_probs, test_probs]),
                        "label": np.concatenate([y_val, y_test]),
                        "feature_base": base,
                        "task": task["name"],
                        "fold": fold_id,
                    }
                )
                pred_out = os.path.join(
                    OUTPUT_DIR,
                    f"predictions_fold{fold_id}_{task['name']}_{base.replace('/', '_')}.csv",
                )
                pred_df.to_csv(pred_out, index=False, encoding="utf-8-sig")

    if records:
        metrics_df = pd.DataFrame(records)
        metrics_path = os.path.join(OUTPUT_DIR, "single_feature_spline_gradient_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
        print(f"[Done] 指标已保存到: {metrics_path}")
    else:
        print("[Warning] 未生成任何指标，可能是训练失败或数据不足。")


if __name__ == "__main__":
    run_single_feature_models()
