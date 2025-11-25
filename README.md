# HCCpre
肝硬化患者预测未来一年发生HCC（ patient with cirrhosis  predicted to develop HCC within the next year）

## 纵向轨迹建模组件

`trajectory_and_risk_models.py` 提供 3 种高级轨迹提取器与 3 种风险预测器，可按 `轨迹提取器 × 风险预测器` 的方式自由组合，支持 “训练使用 T1/T2/T3，推理阶段只输入任意已有时间点” 的需求。

### 轨迹提取器
1. **SplineMixedEffectEncoder**：对每个指标拟合低阶多项式，输出个体化的截距/斜率/曲率系数，模拟混合效应随机项。
2. **FunctionalPCAEncoder**：对所有 `__T1/__T2/__T3` 列执行函数型 PCA，得到平滑后的主成分嵌入。
3. **KalmanTrajectoryEncoder**：对每个指标运行常速度卡尔曼滤波，提取最新的水平与趋势状态作为嵌入。

### 风险预测器
1. **MultiTaskCalibratedLogistic**：共享权重的多任务逻辑回归，针对不同预测时间窗提供独立截距，并输出任务概率。
2. **GradientBoostedRiskModel**：为每个任务分别训练 `HistGradientBoostingClassifier`，捕捉非线性和交互效应。
3. **BayesianRiskAggregator**：基于拉普拉斯近似的贝叶斯逻辑回归，可返回风险概率及不确定性区间。

所有类都遵循 scikit-learn 风格接口，可直接与现有的 5-fold 划分、评估与模型管理脚本对接。

## 特征准备脚本

`prepare_feature_datasets.py` 会：
1. 读取 `/content/drive/.../最终数据时间颠倒版.csv`，识别 59 个实验室指标的 `__T1/__T2/__T3` 列并写入 `base_feature_list.txt`。
2. 与 “病案诊断（第一诊断）.csv” 合并人口学信息，生成 `age__T1`、`sex__T1`。
3. 仅保留 `age__T1`、`sex__T1` 以及所有实验室指标在三个时间点的原始列，执行 Z-Score 标准化，输出 `processed_ML_data.csv` 与 `feature_scaler.joblib`。

脚本不再构造 `Delta` 差分列，方便纵向轨迹模型在训练阶段吸收完整 T1/T2/T3 信息的同时，推理阶段即使只有 T1 观测也能落入同一特征空间。

## 纵向判别训练脚本

`train_longitudinal_models.py` 会：

1. 读取 `processed_ML_data.csv` 与 `feature_scaler.joblib`，反标准化得到原始的 `__T1/__T2/__T3` 列，再结合 `base_feature_list.txt` 中的 59 个实验室指标。
2. 载入 `folds/` 目录下既有的 5 折 (train/val/test) 索引，并在 `第二次训练纵向推理模型/` 目录下对 `3 × 3` 轨迹提取器/风险预测器组合逐一训练，支持断点续跑与 `tqdm` 进度条。
3. 对每个组合输出：
   - `metrics.json`：包含 AUC、时间依赖 AUC、C-index、灵敏度/特异度、F1、PPV、NPV、Brier/时变 Brier、校准曲线等指标；
   - `predictions.csv`：逐样本记录 `subject_id`、fold、split、task、预测概率、阈值化标签、真实标签及时间戳，便于后续在 R 中绘制各种曲线；
   - `model.joblib`：保存 `轨迹提取器 + 风险预测器` 的组合模型；
   - `run_manifest.csv`：在根目录累积各组合的运行时间、验证/测试 AUC，方便快速筛选最优配置。

所有输出均存放在 `/content/drive/MyDrive/Github项目/HCC预测机器学习/第二次训练纵向推理模型` 目录下，便于团队成员直接在同一路径读取结果并进行绘图分析。需要注意的是，脚本中的 C 指数专为固定时间窗（二分类）任务定制，等价于该任务的 ROC-AUC，避免了“事件更早=风险更高”这类不适用于本项目的假设。

> **T1 推理保证**：训练阶段仍使用真实的 `__T1/__T2/__T3` 观测拟合轨迹提取器；而在生成验证/测试预测时，脚本会把所有 `__T2/__T3` 列掩码为 `NaN`，由轨迹提取器内部仅凭 T1 观测推断嵌入，确保推理阶段不会接触任何真实的后续检查结果。

> **Colab/Notebook 小提示**：如果直接将 `train_longitudinal_models.py` 的内容粘贴到单个代码单元中运行，脚本会在无法导入 `trajectory_and_risk_models.py` 时自动加载同样的内联定义，因此无需额外手动 `import` 其它模块即可完成训练。

## 指标后处理脚本

`postprocess_metrics.py` 会遍历 `第二次训练纵向推理模型/fold_*/*/` 目录，读取每个组合生成的 `predictions.csv`，在不重新训练模型的前提下：

1. 依据验证集重新寻找最佳阈值，并用最新版的 C 指数/时间依赖 AUC/Brier 公式计算所有基础与高阶指标；
2. 将结果写入 `metrics_new.json`（不会覆盖原始 `metrics.json`）；
3. 把所有模型、所有折、所有任务/数据集的标量指标整合成单表 `metrics_aggregate_new.csv`，方便在 R 里一次性载入绘图。

这样既能修正历史指标，又能为后续可视化提供统一的原材料。

## 特征重要性分析脚本

`feature_importance_analysis.py` 直接读取 `processed_ML_data.csv` 与 `folds/` 索引，只保留带有 `__T1` 后缀的特征，通过以下步骤生成 “仅 T1 输入” 场景下 `T2`/`T3` 的重要性排名：

1. 在每个折上使用 `HistGradientBoostingClassifier` 拟合 `T2outcome` 与 `T3outcome`；
2. 采用验证集的 permutation importance（`scoring='roc_auc'`、`n_repeats=10`）量化每个特征的贡献；
3. 将“每折 × 每特征”的结果写入 `/content/drive/.../feature_importance/permutation_importance_by_fold.csv`，便于进一步做统计检验或可视化；
4. 同时输出跨折平均值与标准差 `feature_ranking_summary.csv`，脚本运行结束会在终端打印各任务的 Top-15 特征，方便快速挑选或做可解释性汇报。

该脚本沿用与训练流水线完全一致的固定路径，运行 `python feature_importance_analysis.py` 即可复现全部结果。

### 基于 SHAP 的重要性分析

如果需要结合可解释 AI 工具量化特征贡献，可运行 `shap_feature_importance.py`（同样只分析 `__T1` 特征）：

1. 每个折分别为 `T2outcome`、`T3outcome` 训练 `HistGradientBoostingClassifier`；
2. 以训练集采样（最多 200 条）作为 SHAP 背景，调用 `shap.Explainer` 计算验证集的 SHAP 值；
3. 输出 `feature_importance/shap/shap_importance_by_fold.csv`（每折 × 每特征的绝对值平均）以及 `shap_feature_ranking_summary.csv`（跨折平均排名），脚本会在终端打印两个任务的 Top-15 SHAP 特征；
4. 若环境中尚未安装 `shap`，先执行 `pip install shap`。

命令行：

```bash
python shap_feature_importance.py
```

即可得到与 permutation 版本互补的 SHAP 排名。

## 单特征逻辑回归基线

如果需要评估“单个指标独立预测 T2/T3” 的效果，可运行 `single_feature_lr.py`：

1. 仅使用 `processed_ML_data.csv` 中的 `scaled_*__T1` 特征（含 age/sex），保持推理阶段严格 T1-only；
2. 复用 `folds/` 中的 5 折索引，每个折在 train 上拟合单特征 Logistic Regression，在 val 上用 Youden 指数选阈值，再在 val/test 上输出 AUC、PR-AUC、敏感度、特异度、F1、PPV/NPV、Balanced Accuracy、MCC、Brier 等指标；
3. 每个特征 × 任务 × 折的预测详情写入 `/content/drive/.../单特征逻辑回归/predictions_fold{fold}_{task}_{feature}.csv`，汇总指标表写入 `single_feature_lr_metrics.csv` 便于后续排序/可视化；
4. 运行方式：

```bash
python single_feature_lr.py
```

即可在 `/单特征逻辑回归` 目录下生成全部结果。

## 诊断模型（训练看全特征，推理仅少量指标）

如果需要做“诊断”场景（训练吸收全部 T1/T2/T3 轨迹，外部验证仅依赖少数 8~9 个可用化验项），可运行 `train_diagnostic_spline_gradient.py`：

1. 采用最佳组合 **SplineMixedEffectEncoder + HistGradientBoostingClassifier**；
2. 训练阶段使用所有基准指标的 T1/T2/T3 列构造轨迹嵌入；
3. 在验证/测试阶段把未列入 `AVAILABLE_BASE_FEATURES` 的指标（含 T2/T3）全部掩码为 NaN，确保推理只依赖少数可用化验项；
4. 每折输出 `model.joblib`、`pred_val.csv`、`pred_test.csv`、`metrics.json`，汇总写入 `/content/drive/.../诊断模型/summary.csv`；
5. 运行方式：

```bash
python train_diagnostic_spline_gradient.py
```

如需调整推理可用的化验项，可编辑脚本顶部的 `AVAILABLE_BASE_FEATURES` 列表。

## ROC 可视化脚本

`plot_roc_curves.R` 直接使用 `第二次训练纵向推理模型/fold_*/*/predictions.csv` 中的逐样本概率：

1. 对 9 个模型 × 5 折的验证/测试集分别计算 ROC 曲线，并按任务 `T2`、`T3` 拆成四张图（每张图都有 45 条曲线）；
2. 同时绘制对应的 PR 曲线图，输出 `pr_t{2|3}_{val|test}.png`；
3. 从 `metrics_new.json` 中读取 “时间依赖 AUC” 序列，输出 `time_auc_t{2|3}_{val|test}.png`，用于展示不同截止时间的判别趋势；若某些历史结果尚未生成该字段，脚本会自动退回到 `predictions.csv` 即时计算，确保图形依旧可用；
4. 生成 `calibration_t{2|3}_{val|test}.png`，展示 9 个模型 × 5 折的校准曲线，并额外保存 `calibration_stats.csv`，其中包含 Hosmer-Lemeshow 与 Spiegelhalter 统计量；
5. 计算时间依赖 Brier 曲线并输出 `brier_t{2|3}_{val|test}.png`，若 `metrics_new.json` 缺少该字段会自动退回到 `predictions.csv` 实时计算，同时将积分 Brier (IBS) 汇总到 `brier_ibs.csv`；
6. 对验证/测试集扫描 0～1 的阈值，计算 Sensitivity / Specificity / PPV / NPV / F1 与阈值的关系，生成 `threshold_{metric}_t{2|3}_{val|test}.png` 共 20 张图，并把所有数据写入 `threshold_metrics.csv`，便于在 R 中进一步选择部署阈值；
7. 读取 `metrics_aggregate_new.csv` 中的 `Balanced_Accuracy` 与 `MCC`，为 9 个模型 × 5 折绘制小提琴+箱线分布图（T2/T3 × 验证/测试 四个面板），并导出底层数据 `distribution_mcc_balanced_accuracy.csv`；
8. 基于验证/测试集的逐样本概率与真实标签计算 Decision Curve Analysis (Net Benefit)，自动附加 `Treat All` / `Treat None` 参考曲线，输出 `decision_curve_t{2|3}_{val|test}.png` 及 `decision_curve_data.csv`，帮助临床团队评估不同阈值下的净收益；
9. 以 `subject_id` 为键对齐 9 个模型的逐样本概率，计算每个任务在验证/测试集上的 Net Reclassification Improvement (NRI) 与 Integrated Discrimination Improvement (IDI)，并将结果按“基准模型 vs 对比模型”的形式绘制热力图（`heatmap_{nri|idi}_t{2|3}_{val|test}.png`），同时把所有折的平均值写入 `nri_idi_heatmap_data.csv`；
10. 对 `BayesianRiskAggregator` 的验证/测试集结果绘制概率置信区间带（`bayesian_ci_t{2|3}_{val|test}.png`），并把按轨迹提取器聚合后的 `prob_lower/prob_upper` 数据写入 `bayesian_ci_summary.csv`，便于观察模型不确定性；
11. 基于 `metrics_aggregate_new.csv` 中的多项指标绘制 `T2` 与 `T3` 的任务雷达图（`task_radar_{val|test}.png`），同步导出 `task_radar_summary.csv`，帮助快速比较两类预测任务在 AUC、敏感度、特异度、PPV、NPV、F1 与 Balanced Accuracy 上的平均表现；
12. 自动把所有图片写入 `plots/` 子目录；
13. 如果脚本检测到默认目录 `K:/研二/肝硬化 实验室资料 背景文献和原始数据/原始数据/第二次训练纵向推理模型` 已存在，直接运行 `Rscript plot_roc_curves.R` 即可；否则像下面这样手动传入路径：
   ```bash
   Rscript plot_roc_curves.R "K:/研二/肝硬化 实验室资料 背景文献和原始数据/原始数据/第二次训练纵向推理模型"
   ```

这样在本地 R 环境中无需重新连接 Colab，也能一次性得到 ROC、PR、时间依赖 AUC、校准、Brier、阈值扫描、Decision Curve Analysis、MCC/Balanced Accuracy 分布、NRI/IDI 热力图、BayesianRiskAggregator 置信区间以及 T2/T3 任务雷达图等 11 大类图像（外加 Hosmer-Lemeshow/Spiegelhalter 统计表、IBS 汇总、阈值扫描/决策曲线/NRI-IDI/置信区间/雷达原始数据），覆盖 `T2/T3 × 验证/测试` 共 62 张图。
