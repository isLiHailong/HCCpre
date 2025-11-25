"""训练纵向判别模型并输出作图所需的全部指标/预测。"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import dump, load
from tqdm import tqdm

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)

PROJECT_ROOT = "/content/drive/MyDrive/Github项目/HCC预测机器学习"
TIME_SUFFIXES = ("__T1", "__T2", "__T3")
INFERENCE_SPLITS = {"val", "test"}
INFERENCE_ALLOWED_SUFFIXES = ("__T1",)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from trajectory_and_risk_models import (  # noqa: E402
        RISK_PREDICTORS,
        TRAJECTORY_ENCODERS,
    )
except ModuleNotFoundError:
    print(
        "检测到 trajectory_and_risk_models.py 不可用，自动载入内联轨迹/风险组件定义。"
    )
    INLINE_MODEL_DEFINITIONS = r'''
"""Reusable trajectory extractors and risk predictors for longitudinal HCC modeling.

This module provides three advanced trajectory encoders and three risk
predictors that follow scikit-learn style APIs so they can be composed in a
pipeline.  Each trajectory extractor consumes wide-format laboratory data where
each base biomarker has ``__T1``, ``__T2`` and ``__T3`` suffixes.  The
extractors summarise the longitudinal trajectory into a compact embedding that
downstream risk models can ingest while still supporting the "train on all
time-points, infer with partial history" workflow demanded by the project.

All classes are intentionally light-weight so they can run inside the current
Colab-style environment without relying on heavyweight probabilistic
frameworks.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _ensure_times(base_name: str, df: pd.DataFrame) -> List[str]:
    """Return the full set of T1/T2/T3 column names for ``base_name``.

    Raises a ``KeyError`` if any column is missing so issues can be surfaced
    early during feature construction.
    """

    suffixes = ["__T1", "__T2", "__T3"]
    cols = [f"{base_name}{suffix}" for suffix in suffixes]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"缺少 {base_name} 的列: {missing}")
    return cols


def _time_design_matrix(times: Sequence[float], degree: int) -> np.ndarray:
    """Vandermonde design matrix used by spline/polynomial encoders."""

    times = np.asarray(times, dtype=float)
    cols = [times ** k for k in range(degree + 1)]
    return np.vstack(cols).T  # shape: (n_time, degree + 1)


def _stack_tasks(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack multi-task inputs so a shared model can be trained.

    Returns (X_aug, y_vec, task_ids).
    """

    if Y.ndim != 2:
        raise ValueError("Y 必须是 (n_samples, n_tasks) 的二维矩阵")

    n_samples, n_tasks = Y.shape
    task_ids = np.tile(np.arange(n_tasks), n_samples)
    X_aug = np.repeat(X, n_tasks, axis=0)
    y_vec = Y.reshape(-1)
    return X_aug, y_vec, task_ids


# ---------------------------------------------------------------------------
# Trajectory extractors
# ---------------------------------------------------------------------------


class SplineMixedEffectEncoder(TransformerMixin, BaseEstimator):
    """Approximate mixed-effect trajectories via polynomial spline coefficients.

    The encoder fits a low-degree polynomial to each biomarker trajectory per
    subject.  The resulting coefficients (intercept, slope, curvature, ...)
    serve as subject-specific random-effect proxies.  Because coefficients are
    derived analytically, the encoder can process partial histories (e.g. only
    T1) during inference by solving a least-squares system with the available
    time points.
    """

    def __init__(
        self,
        base_features: Optional[Iterable[str]] = None,
        degree: int = 2,
        time_points: Sequence[float] = (0.0, 1.0, 2.0),
    ) -> None:
        self.base_features = None if base_features is None else list(base_features)
        self.degree = degree
        self.time_points = tuple(time_points)

    def fit(self, df: pd.DataFrame, y: Optional[ArrayLike] = None):
        if self.base_features is None:
            self.base_features_ = sorted(
                {
                    col.split("__T")[0]
                    for col in df.columns
                    if col.endswith("__T1") or col.endswith("__T2") or col.endswith("__T3")
                }
            )
        else:
            self.base_features_ = list(self.base_features)
        self.design_ = _time_design_matrix(self.time_points, self.degree)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "design_")

        embeddings: List[np.ndarray] = []
        for base_name in self.base_features_:
            cols = _ensure_times(base_name, df)
            values = df[cols].to_numpy(dtype=float)
            per_sample: List[np.ndarray] = []
            for row in values:
                mask = ~np.isnan(row)
                if not np.any(mask):
                    per_sample.append(np.zeros(self.degree + 1))
                    continue
                design_sub = self.design_[mask]
                target = row[mask]
                coeff_vec, *_ = np.linalg.lstsq(design_sub, target, rcond=None)
                per_sample.append(coeff_vec)
            embeddings.append(np.vstack(per_sample))
        return np.hstack(embeddings)

    def get_feature_names_out(self) -> List[str]:
        check_is_fitted(self, "design_")
        names = []
        for base_name in self.base_features_:
            for k in range(self.degree + 1):
                names.append(f"{base_name}__coef_{k}")
        return names


class FunctionalPCAEncoder(TransformerMixin, BaseEstimator):
    """Functional PCA over smoothed laboratory trajectories."""

    def __init__(self, n_components: int = 20):
        self.n_components = n_components
        self.scaler_ = StandardScaler()
        self.pca_ = PCA(n_components=n_components, random_state=42)

    def fit(self, df: pd.DataFrame, y: Optional[ArrayLike] = None):
        matrix = self._stack_times(df)
        self.scaler_.fit(matrix)
        self.pca_.fit(self.scaler_.transform(matrix))
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        matrix = self._stack_times(df)
        scaled = self.scaler_.transform(matrix)
        return self.pca_.transform(scaled)

    def _stack_times(self, df: pd.DataFrame) -> np.ndarray:
        cols = sorted([c for c in df.columns if c.endswith(("__T1", "__T2", "__T3"))])
        if not cols:
            raise ValueError("数据集中没有时间序列列 (以 __T1/__T2/__T3 结尾)")
        data = df[cols].copy()
        for col in cols:
            if col.endswith("__T1"):
                continue
            base = col.rsplit("__", 1)[0]
            t1_col = f"{base}__T1"
            if t1_col in data.columns:
                mask = data[col].isna()
                if mask.any():
                    data.loc[mask, col] = data.loc[mask, t1_col]
        return data.to_numpy(dtype=float)

    def get_feature_names_out(self) -> List[str]:
        return [f"fpca_component_{i}" for i in range(self.n_components)]


class KalmanTrajectoryEncoder(TransformerMixin, BaseEstimator):
    """Encode trajectories via a constant-velocity Kalman filter per biomarker."""

    def __init__(
        self,
        process_var: float = 0.05,
        obs_var: float = 1.0,
        time_step: float = 1.0,
    ) -> None:
        self.process_var = process_var
        self.obs_var = obs_var
        self.time_step = time_step

    def fit(self, df: pd.DataFrame, y: Optional[ArrayLike] = None):
        self.base_features_ = sorted(
            {
                col.split("__T")[0]
                for col in df.columns
                if col.endswith("__T1") or col.endswith("__T2") or col.endswith("__T3")
            }
        )
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "base_features_")

        embeddings: List[np.ndarray] = []
        for base in self.base_features_:
            cols = _ensure_times(base, df)
            obs = df[cols].to_numpy(dtype=float)
            states = [self._kalman(obs_row) for obs_row in obs]
            embeddings.append(np.vstack(states))
        return np.hstack(embeddings)

    def _kalman(self, obs: np.ndarray) -> np.ndarray:
        # 2D state: [level, trend]
        F = np.array([[1, self.time_step], [0, 1]])
        Q = self.process_var * np.array([[self.time_step ** 4 / 4, self.time_step ** 3 / 2],
                                         [self.time_step ** 3 / 2, self.time_step ** 2]])
        H = np.array([[1, 0]])
        R = np.array([[self.obs_var]])

        state = np.zeros((2, 1))
        cov = np.eye(2)

        for z in obs:
            # prediction
            state = F @ state
            cov = F @ cov @ F.T + Q
            if np.isnan(z):
                continue
            # update
            z_vec = np.array([[z]])
            S = H @ cov @ H.T + R
            K = cov @ H.T @ np.linalg.inv(S)
            y = z_vec - H @ state
            state = state + K @ y
            cov = (np.eye(2) - K @ H) @ cov
        return state.ravel()

    def get_feature_names_out(self) -> List[str]:
        check_is_fitted(self, "base_features_")
        names = []
        for base in self.base_features_:
            names.append(f"{base}__level")
            names.append(f"{base}__trend")
        return names


# ---------------------------------------------------------------------------
# Risk predictors
# ---------------------------------------------------------------------------


class MultiTaskCalibratedLogistic(BaseEstimator, ClassifierMixin):
    """Shared-weight logistic regression with task-specific intercepts."""

    def __init__(self, C: float = 1.0, penalty: str = "l2"):
        self.C = C
        self.penalty = penalty

    def fit(self, X: ArrayLike, Y: ArrayLike):
        X = check_array(X)
        Y = np.asarray(Y)
        X_aug, y_vec, task_ids = _stack_tasks(X, Y)
        task_one_hot = np.zeros((len(task_ids), Y.shape[1]))
        task_one_hot[np.arange(len(task_ids)), task_ids] = 1.0
        design = np.hstack([X_aug, task_one_hot])
        self.model_ = LogisticRegression(C=self.C, penalty=self.penalty, solver="liblinear")
        self.model_.fit(design, y_vec)
        self.n_tasks_ = Y.shape[1]
        self.n_features_ = X.shape[1]
        return self

    def predict_proba(self, X: ArrayLike) -> Dict[str, np.ndarray]:
        check_is_fitted(self, "model_")
        X = check_array(X)
        probas = {}
        for task in range(self.n_tasks_):
            task_cols = np.zeros((len(X), self.n_tasks_))
            task_cols[:, task] = 1.0
            design = np.hstack([X, task_cols])
            probas[f"task_{task}"] = self.model_.predict_proba(design)[:, 1]
        return probas


class GradientBoostedRiskModel(BaseEstimator, ClassifierMixin):
    """Train separate HistGradientBoosting classifiers for each task."""

    def __init__(self, max_depth: int = 3, learning_rate: float = 0.1, max_iter: int = 200):
        from sklearn.ensemble import HistGradientBoostingClassifier

        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self._hgb_cls = HistGradientBoostingClassifier

    def fit(self, X: ArrayLike, Y: ArrayLike):
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

    def predict_proba(self, X: ArrayLike) -> Dict[str, np.ndarray]:
        check_is_fitted(self, "models_")
        X = check_array(X)
        return {f"task_{i}": model.predict_proba(X)[:, 1] for i, model in enumerate(self.models_)}


class BayesianRiskAggregator(BaseEstimator, ClassifierMixin):
    """Laplace-approximated Bayesian logistic regression per task."""

    def __init__(self, prior_precision: float = 1.0, max_iter: int = 50, tol: float = 1e-6):
        self.prior_precision = prior_precision
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X: ArrayLike, Y: ArrayLike):
        X = check_array(X)
        Y = np.asarray(Y)
        self.weights_: List[np.ndarray] = []
        self.covariances_: List[np.ndarray] = []
        for col in range(Y.shape[1]):
            w, cov = self._fit_single(X, Y[:, col])
            self.weights_.append(w)
            self.covariances_.append(cov)
        return self

    def _fit_single(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_features = X.shape[1]
        w = np.zeros(n_features)
        A = self.prior_precision * np.eye(n_features)
        for _ in range(self.max_iter):
            logits = X @ w
            probs = 1 / (1 + np.exp(-logits))
            gradient = X.T @ (y - probs) - A @ w
            W = probs * (1 - probs)
            hessian = -(X.T * W) @ X - A
            step = np.linalg.solve(hessian, gradient)
            w_new = w - step
            if np.linalg.norm(w_new - w) < self.tol:
                w = w_new
                break
            w = w_new
        cov = np.linalg.inv(-(X.T * (probs * (1 - probs))) @ X - A)
        return w, cov

    def predict_proba(self, X: ArrayLike) -> Dict[str, np.ndarray]:
        check_is_fitted(self, "weights_")
        X = check_array(X)
        probas = {}
        for idx, w in enumerate(self.weights_):
            logits = X @ w
            probas[f"task_{idx}"] = 1 / (1 + np.exp(-logits))
        return probas

    def predictive_interval(self, X: ArrayLike, alpha: float = 0.05) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Return mean +/- z * std intervals for each task."""

        from scipy.stats import norm

        check_is_fitted(self, "covariances_")
        X = check_array(X)
        intervals = {}
        z = norm.ppf(1 - alpha / 2)
        for idx, (w, cov) in enumerate(zip(self.weights_, self.covariances_)):
            mean = X @ w
            var = np.sum((X @ cov) * X, axis=1)
            std = np.sqrt(np.maximum(var, 1e-12))
            lower = 1 / (1 + np.exp(-(mean - z * std)))
            upper = 1 / (1 + np.exp(-(mean + z * std)))
            intervals[f"task_{idx}"] = (lower, upper)
        return intervals


# Convenience mapping so notebooks can import by name
TRAJECTORY_ENCODERS: Dict[str, BaseEstimator] = {
    "spline_mixed": SplineMixedEffectEncoder,
    "functional_pca": FunctionalPCAEncoder,
    "kalman_state": KalmanTrajectoryEncoder,
}

RISK_PREDICTORS: Dict[str, BaseEstimator] = {
    "multitask_logistic": MultiTaskCalibratedLogistic,
    "gradient_boosted": GradientBoostedRiskModel,
    "bayesian_laplace": BayesianRiskAggregator,
}


    '''
    exec(INLINE_MODEL_DEFINITIONS, globals())

PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "processed_ML_data.csv")
SCALER_PATH = os.path.join(PROJECT_ROOT, "feature_scaler.joblib")
FOLDS_DIR = os.path.join(PROJECT_ROOT, "folds")
BASE_FEATURES_PATH = os.path.join(PROJECT_ROOT, "base_feature_list.txt")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "第二次训练纵向推理模型")

TRAJECTORY_CHOICES = ["spline_mixed", "functional_pca", "kalman_state"]
RISK_CHOICES = ["multitask_logistic", "gradient_boosted", "bayesian_laplace"]

TASKS = (
    {"name": "T2", "label": "T2outcome", "time": "T2time"},
    {"name": "T3", "label": "T3outcome", "time": "T3time"},
)


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


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def compute_embedding_fill_values(matrix: np.ndarray) -> np.ndarray:
    """Compute per-dimension fallback values for encoder embeddings."""

    if matrix.ndim != 2:
        raise ValueError("embedding 矩阵必须是二维")
    sanitized = matrix.copy()
    sanitized[~np.isfinite(sanitized)] = np.nan
    means = np.nanmean(sanitized, axis=0)
    means = np.where(np.isnan(means), 0.0, means)
    return means


def apply_embedding_fill(matrix: np.ndarray, fill_values: np.ndarray, *, label: str = "") -> np.ndarray:
    """Replace NaN/inf entries using precomputed fill values."""

    if matrix.shape[1] != len(fill_values):
        raise ValueError("embedding 维度与填充值长度不一致")
    mask = ~np.isfinite(matrix)
    if not np.any(mask):
        return matrix
    if label:
        print(
            f"[Warning] {label} 中检测到 {mask.sum()} 个非有限 embedding 值，使用训练均值进行填充。"
        )
    matrix = matrix.copy()
    rows, cols = np.where(mask)
    matrix[rows, cols] = fill_values[cols]
    return matrix


def load_base_features() -> List[str]:
    with open(BASE_FEATURES_PATH, "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


def fill_feature_missing(feature_df: pd.DataFrame) -> pd.DataFrame:
    """填补原始特征矩阵中的缺失值，确保轨迹提取器输入稳定。"""

    missing_total = int(feature_df.isna().sum().sum())
    if missing_total == 0:
        return feature_df

    col_means = feature_df.mean(skipna=True)
    col_means = col_means.fillna(0.0)
    print(
        f"[Info] 检测到 {missing_total} 个缺失值，已使用对应列的均值进行填补。"
    )
    return feature_df.fillna(col_means)


def expand_static_timepoints(feature_df: pd.DataFrame) -> pd.DataFrame:
    """为年龄/性别等静态特征补齐 T2/T3 列，避免轨迹提取器报错。"""

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


def enforce_available_history(
    feature_df: pd.DataFrame, allowed_suffixes: Tuple[str, ...] = INFERENCE_ALLOWED_SUFFIXES
) -> pd.DataFrame:
    """Mask future time points (default: T2/T3) so inference only sees T1.

    纵向编码器依赖固定列名，因此我们不能直接删除 ``__T2``/``__T3`` 列。这里选择将
    这些列的取值置为 ``NaN``，再由编码器内部根据可用的观测（例如仅 T1）计算嵌入，
    保证推理阶段不会触及真实的后续检查结果。
    """

    allowed = set(allowed_suffixes)
    df = feature_df.copy()
    for suffix in TIME_SUFFIXES:
        if suffix in allowed:
            continue
        cols = [c for c in df.columns if c.endswith(suffix)]
        for col in cols:
            df[col] = np.nan
    return df


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
    meta_df = df[["subject_id", "T2outcome", "T3outcome", "T2time", "T3time"]].copy()
    return feature_df, meta_df


def load_fold_indices(fold_dir: str) -> Dict[str, np.ndarray]:
    splits = {}
    for split in ("train", "val", "test"):
        csv_path = os.path.join(fold_dir, f"{split}.csv")
        splits[split] = read_smart(csv_path)["idx"].to_numpy(dtype=int)
    return splits


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


def concordance_index(
    times: np.ndarray, events: np.ndarray, scores: np.ndarray
) -> float:
    """Compute a binary concordance index that matches fixed-horizon tasks.

    在本项目中，T2/T3 标签代表“是否在指定窗口内发生 HCC”，而不是
    “事件在何时发生”。因此这里不再用生存分析里“谁更早谁风险更高”的
    假设，而是将 C 指数退化为区分阳性/阴性的能力——即与 ROC-AUC 一致。

    这样既能保证含义清晰，也能避免时间戳噪声导致的矛盾读数。
    """

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


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    times: np.ndarray,
) -> Dict:
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


@dataclass
class RunContext:
    fold: int
    trajectory_key: str
    risk_key: str
    base_features: List[str]
    feature_df: pd.DataFrame
    meta_df: pd.DataFrame


def instantiate_trajectory(key: str, base_features: List[str]):
    cls = TRAJECTORY_ENCODERS[key]
    if key == "spline_mixed":
        return cls(base_features=base_features, degree=2)
    if key == "functional_pca":
        n_comp = min(60, max(10, len(base_features)))
        return cls(n_components=n_comp)
    return cls()


def instantiate_risk(key: str):
    cls = RISK_PREDICTORS[key]
    if key == "gradient_boosted":
        return cls(max_depth=3, learning_rate=0.05, max_iter=500)
    if key == "bayesian_laplace":
        return cls(prior_precision=1.0, max_iter=100)
    return cls()


def append_manifest(row: Dict[str, object]) -> None:
    manifest_path = os.path.join(RESULTS_DIR, "run_manifest.csv")
    df = pd.DataFrame([row])
    header = not os.path.exists(manifest_path)
    df.to_csv(manifest_path, mode="a", header=header, index=False, encoding="utf-8-sig")


def run_training() -> None:
    ensure_dir(RESULTS_DIR)
    feature_df, meta_df = load_feature_frame()
    base_features = load_base_features()
    total = len(TRAJECTORY_CHOICES) * len(RISK_CHOICES) * 5
    pbar = tqdm(total=total, desc="纵向判别训练")
    for fold in range(1, 6):
        fold_dir = os.path.join(FOLDS_DIR, f"fold_{fold}")
        splits = load_fold_indices(fold_dir)
        for traj_key in TRAJECTORY_CHOICES:
            for risk_key in RISK_CHOICES:
                run_dir = os.path.join(RESULTS_DIR, f"fold_{fold}", f"{traj_key}__{risk_key}")
                ensure_dir(run_dir)
                metrics_path = os.path.join(run_dir, "metrics.json")
                if os.path.exists(metrics_path):
                    pbar.set_description(f"跳过 fold{fold}-{traj_key}+{risk_key}")
                    pbar.update(1)
                    continue
                ctx = RunContext(
                    fold=fold,
                    trajectory_key=traj_key,
                    risk_key=risk_key,
                    base_features=base_features,
                    feature_df=feature_df,
                    meta_df=meta_df,
                )
                result = execute_run(ctx, splits)
                with open(metrics_path, "w", encoding="utf-8") as fh:
                    json.dump(result["metrics"], fh, ensure_ascii=False, indent=2, cls=NpEncoder)
                result["predictions"].to_csv(
                    os.path.join(run_dir, "predictions.csv"),
                    index=False,
                    encoding="utf-8-sig",
                )
                dump(result["model"], os.path.join(run_dir, "model.joblib"))
                append_manifest(result["manifest_row"])
                pbar.set_description(f"完成 fold{fold}-{traj_key}+{risk_key}")
                pbar.update(1)
    pbar.close()


def execute_run(ctx: RunContext, splits: Dict[str, np.ndarray]):
    encoder = instantiate_trajectory(ctx.trajectory_key, ctx.base_features)
    X_train_df = ctx.feature_df.iloc[splits["train"]].reset_index(drop=True)
    start = time.time()
    encoder.fit(X_train_df)
    X_train = encoder.transform(X_train_df)
    fill_values = compute_embedding_fill_values(X_train)
    X_train = apply_embedding_fill(X_train, fill_values, label=f"fold{ctx.fold}-train")
    Y_train = np.column_stack(
        [ctx.meta_df.loc[splits["train"], task["label"]].to_numpy() for task in TASKS]
    )
    risk_model = instantiate_risk(ctx.risk_key)
    risk_model.fit(X_train, Y_train)
    runtime = time.time() - start

    probabilities: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    pred_rows: List[pd.DataFrame] = []
    supports_interval = hasattr(risk_model, "predictive_interval")

    for split_name, idx in splits.items():
        split_df = ctx.feature_df.iloc[idx].reset_index(drop=True)
        if split_name in INFERENCE_SPLITS:
            split_df = enforce_available_history(split_df)
        X_split = encoder.transform(split_df)
        X_split = apply_embedding_fill(
            X_split,
            fill_values,
            label=f"fold{ctx.fold}-{ctx.trajectory_key}-{split_name}",
        )
        prob_dict = risk_model.predict_proba(X_split)
        interval_dict: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        if supports_interval:
            try:
                interval_dict = risk_model.predictive_interval(X_split)  # type: ignore[attr-defined]
            except Exception as exc:  # pragma: no cover - diagnostic logging only
                print(
                    f"[Warning] predictive_interval failed for fold {ctx.fold}"
                    f" {ctx.trajectory_key}+{ctx.risk_key} on {split_name}: {exc}"
                )
                interval_dict = {}
        for task_idx, task in enumerate(TASKS):
            y_true = ctx.meta_df.loc[idx, task["label"]].to_numpy()
            times = ctx.meta_df.loc[idx, task["time"]].to_numpy()
            probs = prob_dict[f"task_{task_idx}"]
            interval_key = f"task_{task_idx}"
            prob_lower = np.full_like(probs, np.nan, dtype=float)
            prob_upper = np.full_like(probs, np.nan, dtype=float)
            if interval_key in interval_dict:
                lower_arr, upper_arr = interval_dict[interval_key]
                prob_lower = lower_arr
                prob_upper = upper_arr
            probabilities.setdefault(task["name"], {})[split_name] = {
                "y_true": y_true,
                "y_prob": probs,
                "times": times,
            }
            pred_rows.append(
                pd.DataFrame(
                    {
                        "subject_id": ctx.meta_df.loc[idx, "subject_id"].to_numpy(),
                        "fold": ctx.fold,
                        "split": split_name,
                        "task": task["name"],
                        "probability": probs,
                        "prob_lower": prob_lower,
                        "prob_upper": prob_upper,
                        "label": y_true,
                        "time": times,
                    }
                )
            )
    pred_df = pd.concat(pred_rows, ignore_index=True)

    metrics_payload = {
        "fold": ctx.fold,
        "trajectory": ctx.trajectory_key,
        "risk": ctx.risk_key,
        "runtime_seconds": runtime,
        "generated_at": datetime.utcnow().isoformat(),
        "tasks": {},
    }

    thresholds: Dict[str, float] = {}
    for task in TASKS:
        val_entry = probabilities[task["name"]]["val"]
        thr = find_best_threshold(val_entry["y_true"], val_entry["y_prob"])
        thresholds[task["name"]] = thr
        metrics_payload["tasks"][task["name"]] = {"threshold": thr, "splits": {}}
        for split_name in ("train", "val", "test"):
            entry = probabilities[task["name"]][split_name]
            metrics_payload["tasks"][task["name"]]["splits"][split_name] = compute_metrics(
                entry["y_true"], entry["y_prob"], thr, entry["times"]
            )

    for task_name, thr in thresholds.items():
        mask = pred_df["task"] == task_name
        pred_df.loc[mask, "threshold"] = thr
        pred_df.loc[mask, "pred_label"] = (pred_df.loc[mask, "probability"] >= thr).astype(int)

    manifest_row = {
        "timestamp": datetime.utcnow().isoformat(),
        "fold": ctx.fold,
        "trajectory": ctx.trajectory_key,
        "risk": ctx.risk_key,
        "runtime_sec": runtime,
    }
    for task in TASKS:
        for split_name in ("val", "test"):
            key = f"{task['name']}_{split_name}_auc"
            value = metrics_payload["tasks"][task["name"]]["splits"][split_name].get("AUC")
            manifest_row[key] = value

    model_bundle = {
        "encoder": encoder,
        "risk_model": risk_model,
        "embedding_fill_values": fill_values,
    }
    return {"metrics": metrics_payload, "predictions": pred_df, "model": model_bundle, "manifest_row": manifest_row}


if __name__ == "__main__":
    run_training()
