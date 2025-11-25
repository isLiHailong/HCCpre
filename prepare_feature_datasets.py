"""生成纵向判别用的基础特征矩阵。

该脚本严格沿用原始 Colab 工程中的固定路径，只完成两件事：
1. 从纵向实验室表里识别 59 个实验室指标（T1/T2/T3），并与诊断表合并获取 age/sex；
2. 输出仅包含 ``age__T1``、``sex__T1`` 以及所有 ``feature__T{1,2,3}`` 的标准化矩阵，
   供后续“轨迹提取器 + 风险预测器”直接消费。

注意：本脚本不再构造任何 ``Delta`` 差分特征，确保训练虽然吸收三次时间点，
推理阶段仍可以只提供 T1 数据即可落在同一特征空间。
"""

from __future__ import annotations

import os
import re
import warnings
from typing import Iterable

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = "/content/drive/MyDrive/Github项目/HCC预测机器学习"
TIME_DATA_CSV = "/content/drive/MyDrive/附三实验室资料/最终数据时间颠倒版.csv"
DIAG_CSV = "/content/drive/MyDrive/附三实验室资料/病案诊断（第一诊断）.csv"
BASE_FEATURES_PATH = os.path.join(PROJECT_ROOT, "base_feature_list.txt")
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "processed_ML_data.csv")
SCALER_PATH = os.path.join(PROJECT_ROOT, "feature_scaler.joblib")

os.makedirs(PROJECT_ROOT, exist_ok=True)


def read_smart(path: str) -> pd.DataFrame:
    for enc in ("utf-8-sig", "gb18030", "gbk", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception:
            continue
    return pd.read_csv(path, low_memory=False)


def pick_col(df: pd.DataFrame, patterns: Iterable[str]) -> str | None:
    for pat in patterns:
        hits = [c for c in df.columns if re.search(pat, str(c), re.I)]
        if hits:
            return hits[0]
    return None


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


def detect_base_features(df: pd.DataFrame) -> list[str]:
    auxiliary_cols = {
        "subject_id",
        "visit_no",
        "T2outcome",
        "T3outcome",
        "T2time",
        "T3time",
        "sex_num",
        "HCC",
        "HCC_from_diag",
    }
    suffixes = ("__T1", "__T2", "__T3")
    features: set[str] = set()
    for col in df.columns:
        if col.lower() in {c.lower() for c in auxiliary_cols}:
            continue
        for suffix in suffixes:
            if col.endswith(suffix):
                base = col[: -len(suffix)]
                if base and base.lower() not in {"age", "sex"}:
                    features.add(base)
                break
    base_feature_list = sorted(features)
    with open(BASE_FEATURES_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(base_feature_list))
    return base_feature_list


def main() -> None:
    print("--- 步骤 1: 加载数据 ---")
    try:
        df = read_smart(TIME_DATA_CSV)
        diag = read_smart(DIAG_CSV)
    except FileNotFoundError as exc:
        raise RuntimeError(f"加载文件失败: {exc}") from exc

    base_features = detect_base_features(df)
    print(f"成功识别 {len(base_features)} 个实验室基准特征。")

    print("--- 步骤 2: 合并人口学信息 ---")
    df = df.rename(columns={"subject_id": "subject_id_main"}, errors="ignore")
    if "subject_id_main" not in df.columns:
        df = df.rename(columns={df.columns[0]: "subject_id_main"}, errors="ignore")
    df["subject_id_main"] = df["subject_id_main"].astype(str)

    visit_col = pick_col(diag, [r"visit.?no", r"就诊", r"住院号", r"门诊号", r"唯一就诊"])
    age_col = pick_col(diag, [r"(^|[^a-z])age([^a-z]|$)", r"年龄"])
    sex_col = pick_col(diag, [r"(^|[^a-z])sex([^a-z]|$)", r"gender", r"性别"])
    if not all((visit_col, age_col, sex_col)):
        raise RuntimeError("诊断表缺少 visit/age/sex 列，无法生成 age/sex 特征。")

    diag_min = diag[[visit_col, age_col, sex_col]].copy()
    diag_min.columns = ["visit_no", "age_raw", "sex_raw"]
    diag_min["visit_no"] = diag_min["visit_no"].astype(str)
    diag_min["gender"] = diag_min["sex_raw"].apply(map_gender)
    diag_min["age"] = pd.to_numeric(diag_min["age_raw"], errors="coerce")

    df = df.merge(
        diag_min[["visit_no", "age", "gender"]],
        left_on="subject_id_main",
        right_on="visit_no",
        how="left",
    )
    df["age__T1"] = df["age"].fillna(method="ffill")
    df["sex__T1"] = df["gender"].fillna(method="ffill")

    print("--- 步骤 3: 构建三次时间点特征列表 ---")
    feature_columns: list[str] = ["age__T1", "sex__T1"]
    for base in base_features:
        cols = [f"{base}__T1", f"{base}__T2", f"{base}__T3"]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise RuntimeError(f"缺少 {base} 的列: {missing}")
        feature_columns.extend(cols)
    print(f"最终纳入 {len(feature_columns)} 个特征 (含三次时间点)。")

    print("--- 步骤 4: Z-Score 标准化 ---")
    X = df[feature_columns].to_numpy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    dump(scaler, SCALER_PATH)
    print(f"StandardScaler 已保存到: {SCALER_PATH}")

    print("--- 步骤 5: 导出特征矩阵 ---")
    scaled_cols = [f"scaled_{col}" for col in feature_columns]
    df_processed = pd.DataFrame(X_scaled, columns=scaled_cols)
    df_processed["subject_id"] = df["subject_id_main"]
    df_processed["T2outcome"] = df["T2outcome"]
    df_processed["T3outcome"] = df["T3outcome"]
    df_processed["T2time"] = df["T2time"]
    df_processed["T3time"] = df["T3time"]
    df_processed.to_csv(PROCESSED_DATA_PATH, index=False, encoding="utf-8-sig")
    print(f"最终矩阵保存到: {PROCESSED_DATA_PATH}")
    print("\n--- 特征工程完成 ---")


if __name__ == "__main__":
    main()
