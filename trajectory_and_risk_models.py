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
            coeffs, _, _, _ = np.linalg.lstsq(self.design_, values.T, rcond=None)
            # coeffs shape: (degree+1, n_samples)
            embeddings.append(coeffs.T)
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
        return df[cols].to_numpy(dtype=float)

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
            z = np.array([[z]])
            # prediction
            state = F @ state
            cov = F @ cov @ F.T + Q
            # update
            S = H @ cov @ H.T + R
            K = cov @ H.T @ np.linalg.inv(S)
            y = z - H @ state
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

