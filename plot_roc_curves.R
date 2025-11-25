#!/usr/bin/env Rscript
# 绘制 3x3 模型在验证/测试集上的 ROC 曲线。
# 使用方式：Rscript plot_roc_curves.R /path/to/第二次训练纵向推理模型

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(purrr)
  library(stringr)
  library(ggplot2)
  library(pROC)
  library(glue)
  library(jsonlite)
  library(forcats)
})

# 默认阈值网格（用于阈值扫描与 DCA）；若用户在交互式环境中
# 单独运行部分代码块，下面的默认序列也能保证函数正常工作。
DEFAULT_DCA_THRESHOLDS <- seq(0.01, 0.99, by = 0.01)
DEFAULT_THRESHOLD_GRID <- seq(0, 1, length.out = 201)

resolve_thresholds <- function(thresholds, default_values, fallback_name = NULL) {
  if (!is.null(thresholds)) {
    return(thresholds)
  }
  if (!is.null(fallback_name) && exists(fallback_name, inherits = TRUE)) {
    return(get(fallback_name, inherits = TRUE))
  }
  default_values
}

# 默认结果目录，若未传入参数且该目录存在，则自动使用。
DEFAULT_RESULTS_DIR <- "K:/研二/肝硬化 实验室资料 背景文献和原始数据/原始数据/第二次训练纵向推理模型"

args <- commandArgs(trailingOnly = TRUE)
results_dir <- if (length(args) >= 1) {
  args[[1]]
} else if (dir.exists(DEFAULT_RESULTS_DIR)) {
  message(glue("未传入参数，自动使用默认目录: {DEFAULT_RESULTS_DIR}"))
  DEFAULT_RESULTS_DIR
} else {
  stop(
    "请提供第二次训练纵向推理模型文件夹路径，例如:\n",
    "  Rscript plot_roc_curves.R 'K:/.../第二次训练纵向推理模型'\n",
    "或将 DEFAULT_RESULTS_DIR 改成你的本地路径。"
  )
}
if (!dir.exists(results_dir)) {
  stop(glue("目录不存在: {results_dir}"))
}

list_prediction_tables <- function(base_dir) {
  fold_dirs <- list.dirs(base_dir, full.names = TRUE, recursive = FALSE)
  fold_dirs <- fold_dirs[grepl("fold_\\d+$", basename(fold_dirs))]
  if (length(fold_dirs) == 0) {
    stop(glue("在 {base_dir} 下未找到 fold_* 子目录"))
  }
  map_dfr(fold_dirs, function(fold_dir) {
    model_dirs <- list.dirs(fold_dir, full.names = TRUE, recursive = FALSE)
    map_dfr(model_dirs, function(model_dir) {
      pred_path <- file.path(model_dir, "predictions.csv")
      if (!file.exists(pred_path)) return(NULL)
      dt <- fread(pred_path, encoding = "UTF-8")
      dt$fold_dir <- basename(fold_dir)
      dt$model_dir <- basename(model_dir)
      dt
    })
  })
}

pred_dt <- list_prediction_tables(results_dir)
if (!all(c("split", "task", "probability", "label") %in% names(pred_dt))) {
  stop("predictions.csv 中缺少 split/task/probability/label 列，无法绘制 ROC")
}

pred_dt <- pred_dt %>%
  mutate(
    split = tolower(split),
    trajectory = str_replace(model_dir, "__.*$", ""),
    risk = str_replace(model_dir, "^.*__", ""),
    model_label = glue("{trajectory} + {risk}"),
    fold = str_extract(fold_dir, "\\d+")
  )

model_levels <- sort(unique(pred_dt$model_label))

compute_roc <- function(df) {
  df <- df %>% filter(!is.na(probability), !is.na(label))
  if (n_distinct(df$label) < 2) return(NULL)
  roc_obj <- pROC::roc(response = df$label, predictor = df$probability, quiet = TRUE, direction = "<")
  tibble(
    fpr = 1 - roc_obj$specificities,
    tpr = roc_obj$sensitivities,
    auc = as.numeric(pROC::auc(roc_obj))
  )
}

roc_dt <- pred_dt %>%
  filter(split %in% c("val", "test")) %>%
  group_by(split, task, fold, model_label) %>%
  group_modify(~{
    roc_tbl <- compute_roc(.x)
    if (is.null(roc_tbl)) return(tibble())
    mutate(roc_tbl, fold = unique(.x$fold), model_label = unique(.x$model_label))
  }) %>%
  ungroup()

if (nrow(roc_dt) == 0) {
  stop("没有可用的 ROC 数据，检查 predictions.csv 是否包含验证/测试集记录")
}

plot_split_task <- function(split_name, task_name) {
  df <- roc_dt %>% filter(split == split_name, task == task_name)
  if (nrow(df) == 0) return(NULL)
  ggplot(df, aes(x = fpr, y = tpr, color = model_label, group = interaction(model_label, fold))) +
    geom_path(alpha = 0.7) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray70") +
    scale_color_brewer(palette = "Set1") +
    labs(
      title = glue("{task_name} - {toupper(split_name)} 集 3x3 模型 ROC 曲线 (45 条曲线)"),
      subtitle = "每条曲线 = 1 个模型 × 1 个折",
      x = "1 - Specificity",
      y = "Sensitivity",
      color = "轨迹提取器 + 风险预测器"
    ) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "bottom", legend.title = element_text(size = 10))
}

output_dir <- file.path(results_dir, "plots")
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

tasks <- sort(unique(roc_dt$task))
splits <- c("val", "test")
for (task_name in tasks) {
  for (split_name in splits) {
    plt <- plot_split_task(split_name, task_name)
    if (is.null(plt)) next
    outfile <- file.path(output_dir, glue("roc_{tolower(task_name)}_{split_name}.png"))
    ggsave(outfile, plt, width = 10, height = 6, dpi = 300)
  }
}

message(glue("ROC 图已输出到: {output_dir}"))

# --- PR脚本曲线 ---

compute_pr <- function(df) {
  df <- df %>% filter(!is.na(probability), !is.na(label))
  if (n_distinct(df$label) < 2) return(NULL)
  df <- df %>% arrange(desc(probability))
  tp <- cumsum(df$label == 1)
  fp <- cumsum(df$label == 0)
  total_pos <- sum(df$label == 1)
  precision <- tp / pmax(tp + fp, 1)
  recall <- tp / total_pos
  tibble(
    recall = c(0, recall),
    precision = c(1, precision)
  )
}

pr_dt <- pred_dt %>%
  filter(split %in% c("val", "test")) %>%
  group_by(split, task, fold, model_label) %>%
  group_modify(~{
    pr_tbl <- compute_pr(.x)
    if (is.null(pr_tbl)) return(tibble())
    mutate(pr_tbl, fold = unique(.x$fold), model_label = unique(.x$model_label))
  }) %>%
  ungroup()

plot_pr_split_task <- function(split_name, task_name) {
  df <- pr_dt %>% filter(split == split_name, task == task_name)
  if (nrow(df) == 0) return(NULL)
  ggplot(df, aes(x = recall, y = precision, color = model_label, group = interaction(model_label, fold))) +
    geom_path(alpha = 0.7) +
    geom_hline(yintercept = mean(pred_dt$label), linetype = "dotted", color = "gray60") +
    scale_color_brewer(palette = "Set1") +
    coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
    labs(
      title = glue("{task_name} - {toupper(split_name)} 集 3x3 模型 PR 曲线 (45 条曲线)"),
      subtitle = "每条曲线 = 1 个模型 × 1 个折",
      x = "Recall",
      y = "Precision",
      color = "轨迹提取器 + 风险预测器"
    ) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "bottom", legend.title = element_text(size = 10))
}

for (task_name in tasks) {
  for (split_name in splits) {
    plt <- plot_pr_split_task(split_name, task_name)
    if (is.null(plt)) next
    outfile <- file.path(output_dir, glue("pr_{tolower(task_name)}_{split_name}.png"))
    ggsave(outfile, plt, width = 10, height = 6, dpi = 300)
  }
}

message(glue("PR 曲线亦已输出到: {output_dir}"))

# --- 时间依赖 ROC / AUC 轨迹 ---

list_time_auc_tables <- function(base_dir) {
  fold_dirs <- list.dirs(base_dir, full.names = TRUE, recursive = FALSE)
  fold_dirs <- fold_dirs[grepl("fold_\\d+$", basename(fold_dirs))]
  if (length(fold_dirs) == 0) {
    return(tibble())
  }
  map_dfr(fold_dirs, function(fold_dir) {
    fold_id <- stringr::str_extract(basename(fold_dir), "\\d+")
    model_dirs <- list.dirs(fold_dir, full.names = TRUE, recursive = FALSE)
    map_dfr(model_dirs, function(model_dir) {
      metrics_path <- file.path(model_dir, "metrics_new.json")
      if (!file.exists(metrics_path)) return(tibble())
      payload <- tryCatch(read_json(metrics_path, simplifyVector = FALSE), error = function(e) NULL)
      if (is.null(payload) || is.null(payload$tasks)) return(tibble())
      model_name <- basename(model_dir)
      trajectory <- str_replace(model_name, "__.*$", "")
      risk <- str_replace(model_name, "^.*__", "")
      model_label <- glue("{trajectory} + {risk}")
      map_dfr(names(payload$tasks), function(task_name) {
        task_info <- payload$tasks[[task_name]]
        splits <- task_info$splits
        if (is.null(splits)) return(tibble())
        map_dfr(names(splits), function(split_name) {
          metrics <- splits[[split_name]]
          if (is.null(metrics)) return(tibble())
          curves <- metrics$Time_Dependent_AUC
          if (is.null(curves) || length(curves) == 0) return(tibble())
          curve_df <- map_dfr(curves, function(entry) {
            tibble(
              time = as.numeric(entry$time),
              auc = as.numeric(entry$auc),
              n_samples = as.numeric(entry$n_samples)
            )
          })
          curve_df %>%
            filter(!is.na(time)) %>%
            mutate(
              split = tolower(split_name),
              task = task_name,
              fold = fold_id,
              model_label = model_label
            )
        })
      })
    })
  })
}

time_auc_dt <- list_time_auc_tables(results_dir)

compute_time_auc_from_predictions <- function(df) {
  df <- df %>% filter(!is.na(time)) %>% arrange(time)
  if (nrow(df) < 10) return(tibble())
  unique_times <- sort(unique(df$time))
  map_dfr(unique_times, function(tau) {
    subset <- df %>% filter(time <= tau)
    if (nrow(subset) < 10) return(tibble())
    auc_val <- NA_real_
    if (n_distinct(subset$label) >= 2) {
      roc_obj <- tryCatch(
        pROC::roc(response = subset$label, predictor = subset$probability, quiet = TRUE, direction = "<"),
        error = function(e) NULL
      )
      if (!is.null(roc_obj)) {
        auc_val <- as.numeric(pROC::auc(roc_obj))
      }
    }
    tibble(time = tau, auc = auc_val, n_samples = nrow(subset))
  })
}

build_time_auc_from_predictions <- function(predictions_dt) {
  predictions_dt %>%
    filter(split %in% c("val", "test"), !is.na(time)) %>%
    mutate(time = as.numeric(time)) %>%
    group_by(split, task, fold, model_label) %>%
    group_modify(~{
      curves <- compute_time_auc_from_predictions(.x)
      if (nrow(curves) == 0) return(tibble())
      mutate(curves, split = unique(.x$split), task = unique(.x$task), fold = unique(.x$fold), model_label = unique(.x$model_label))
    }) %>%
    ungroup()
}

if (nrow(time_auc_dt) == 0) {
  message("未在 metrics_new.json 中读到时间依赖 AUC，改为直接基于 predictions.csv 计算……")
  time_auc_dt <- build_time_auc_from_predictions(pred_dt)
}

plot_time_auc <- function(split_name, task_name) {
  df <- time_auc_dt %>% filter(split == split_name, task == task_name, !is.na(auc))
  if (nrow(df) == 0) return(NULL)
  ggplot(df, aes(x = time, y = auc, color = model_label, group = interaction(model_label, fold))) +
    geom_line(alpha = 0.7) +
    scale_color_brewer(palette = "Set1") +
    labs(
      title = glue("{task_name} - {toupper(split_name)} 集 时间依赖 ROC/AUC 轨迹"),
      subtitle = "每条轨迹 = 1 个模型 × 1 个折，展示截止时间的 AUC",
      x = "时间 (与预测脚本中的 *_time 列一致)",
      y = "Time-dependent AUC",
      color = "轨迹提取器 + 风险预测器"
    ) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "bottom", legend.title = element_text(size = 10))
}

if (nrow(time_auc_dt) > 0) {
  for (task_name in tasks) {
    for (split_name in splits) {
      plt <- plot_time_auc(split_name, task_name)
      if (is.null(plt)) next
      outfile <- file.path(output_dir, glue("time_auc_{tolower(task_name)}_{split_name}.png"))
      ggsave(outfile, plt, width = 10, height = 6, dpi = 300)
    }
  }
  message(glue("时间依赖 AUC 轨迹亦已输出到: {output_dir}"))
} else {
  message("未在 metrics_new.json 中找到时间依赖 AUC 信息，跳过该图。")
}

# --- 校准曲线 + Hosmer-Lemeshow / Spiegelhalter 统计量 ---

compute_calibration_bins <- function(df, max_bins = 10) {
  df <- df %>% filter(!is.na(probability), !is.na(label))
  if (nrow(df) < 20 || n_distinct(df$label) < 2) return(list(bins = tibble(), stats = NULL))
  n_bins <- min(max_bins, nrow(df))
  df <- df %>% mutate(bin = ntile(probability, n_bins))
  bins <- df %>%
    group_by(bin) %>%
    summarise(
      n = n(),
      mean_pred = mean(probability),
      obs_rate = mean(label),
      exp_pos = sum(probability),
      obs_pos = sum(label),
      .groups = "drop"
    ) %>%
    mutate(p_hat = pmin(pmax(exp_pos / n, 1e-6), 1 - 1e-6))
  hl_num <- sum((bins$obs_pos - bins$exp_pos)^2 / (bins$n * bins$p_hat * (1 - bins$p_hat) + 1e-9))
  hl_stat <- min(hl_num, 1e9)
  hl_df <- max(nrow(bins) - 2, 1)
  hl_p <- pchisq(hl_stat, df = hl_df, lower.tail = FALSE)
  diff_sum <- sum(df$label - df$probability)
  diff_var <- sum(df$probability * (1 - df$probability))
  spiegel_z <- if (diff_var > 0) diff_sum / sqrt(diff_var) else NA_real_
  spiegel_p <- if (!is.na(spiegel_z)) 2 * pnorm(-abs(spiegel_z)) else NA_real_
  bins_plot <- bins %>% select(-p_hat)
  list(
    bins = bins_plot,
    stats = tibble(
      hosmer_lemeshow = hl_stat,
      hosmer_df = hl_df,
      hosmer_p = hl_p,
      spiegelhalter_z = spiegel_z,
      spiegelhalter_p = spiegel_p,
      n_bins = nrow(bins)
    )
  )
}

calibration_bins_dt <- pred_dt %>%
  filter(split %in% c("val", "test")) %>%
  group_by(split, task, fold, model_label) %>%
  group_modify(~{
    res <- compute_calibration_bins(.x)
    if (nrow(res$bins) == 0) return(tibble())
    mutate(res$bins,
           split = unique(.x$split),
           task = unique(.x$task),
           fold = unique(.x$fold),
           model_label = unique(.x$model_label))
  }) %>%
  ungroup()

calibration_stats_dt <- pred_dt %>%
  filter(split %in% c("val", "test")) %>%
  group_by(split, task, fold, model_label) %>%
  group_modify(~{
    res <- compute_calibration_bins(.x)
    if (is.null(res$stats)) return(tibble())
    mutate(res$stats,
           split = unique(.x$split),
           task = unique(.x$task),
           fold = unique(.x$fold),
           model_label = unique(.x$model_label))
  }) %>%
  ungroup()

plot_calibration <- function(split_name, task_name) {
  df <- calibration_bins_dt %>% filter(split == split_name, task == task_name)
  if (nrow(df) == 0) return(NULL)
  ggplot(df, aes(x = mean_pred, y = obs_rate, color = model_label, group = interaction(model_label, fold))) +
    geom_point(alpha = 0.8) +
    geom_line(alpha = 0.6) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray60") +
    scale_color_brewer(palette = "Set1") +
    coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
    labs(
      title = glue("{task_name} - {toupper(split_name)} 集 校准曲线"),
      subtitle = "点/线 = 每模型×折的分箱平均值，虚线为理想校准",
      x = "预测概率 (分箱均值)",
      y = "实际发生率",
      color = "轨迹提取器 + 风险预测器"
    ) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "bottom", legend.title = element_text(size = 10))
}

if (nrow(calibration_bins_dt) > 0) {
  for (task_name in tasks) {
    for (split_name in splits) {
      plt <- plot_calibration(split_name, task_name)
      if (is.null(plt)) next
      outfile <- file.path(output_dir, glue("calibration_{tolower(task_name)}_{split_name}.png"))
      ggsave(outfile, plt, width = 10, height = 6, dpi = 300)
    }
  }
  message(glue("校准曲线亦已输出到: {output_dir}"))
} else {
  message("未生成校准曲线（可能是该 split 只有单一标签或样本过少）。")
}

if (nrow(calibration_stats_dt) > 0) {
  stats_path <- file.path(output_dir, "calibration_stats.csv")
  fwrite(calibration_stats_dt, stats_path)
  message(glue("Hosmer-Lemeshow / Spiegelhalter 统计表已保存: {stats_path}"))
}

# --- Brier 曲线与积分 Brier (IBS) ---

list_time_brier_tables <- function(base_dir) {
  fold_dirs <- list.dirs(base_dir, full.names = TRUE, recursive = FALSE)
  fold_dirs <- fold_dirs[grepl("fold_\\d+$", basename(fold_dirs))]
  if (length(fold_dirs) == 0) {
    return(tibble())
  }
  map_dfr(fold_dirs, function(fold_dir) {
    fold_id <- stringr::str_extract(basename(fold_dir), "\\d+")
    model_dirs <- list.dirs(fold_dir, full.names = TRUE, recursive = FALSE)
    map_dfr(model_dirs, function(model_dir) {
      metrics_path <- file.path(model_dir, "metrics_new.json")
      if (!file.exists(metrics_path)) return(tibble())
      payload <- tryCatch(read_json(metrics_path, simplifyVector = FALSE), error = function(e) NULL)
      if (is.null(payload) || is.null(payload$tasks)) return(tibble())
      model_name <- basename(model_dir)
      trajectory <- str_replace(model_name, "__.*$", "")
      risk <- str_replace(model_name, "^.*__", "")
      model_label <- glue("{trajectory} + {risk}")
      map_dfr(names(payload$tasks), function(task_name) {
        task_info <- payload$tasks[[task_name]]
        splits <- task_info$splits
        if (is.null(splits)) return(tibble())
        map_dfr(names(splits), function(split_name) {
          metrics <- splits[[split_name]]
          if (is.null(metrics)) return(tibble())
          curves <- metrics$Time_Dependent_Brier
          if (is.null(curves) || length(curves) == 0) return(tibble())
          curve_df <- map_dfr(curves, function(entry) {
            tibble(
              time = as.numeric(entry$time),
              brier = as.numeric(entry$brier),
              n_samples = as.numeric(entry$n_samples)
            )
          })
          curve_df %>%
            filter(!is.na(time)) %>%
            mutate(
              split = tolower(split_name),
              task = task_name,
              fold = fold_id,
              model_label = model_label
            )
        })
      })
    })
  })
}

time_brier_dt <- list_time_brier_tables(results_dir)

compute_time_brier_from_predictions <- function(df) {
  df <- df %>% filter(!is.na(time)) %>% mutate(time = as.numeric(time)) %>% arrange(time)
  if (nrow(df) < 5) return(tibble())
  unique_times <- sort(unique(df$time))
  map_dfr(unique_times, function(tau) {
    subset <- df %>% filter(time <= tau)
    if (nrow(subset) == 0) return(tibble())
    tibble(
      time = tau,
      brier = mean((subset$label - subset$probability)^2, na.rm = TRUE),
      n_samples = nrow(subset)
    )
  })
}

build_time_brier_from_predictions <- function(predictions_dt) {
  predictions_dt %>%
    filter(split %in% c("val", "test"), !is.na(time)) %>%
    mutate(time = as.numeric(time)) %>%
    group_by(split, task, fold, model_label) %>%
    group_modify(~{
      curves <- compute_time_brier_from_predictions(.x)
      if (nrow(curves) == 0) return(tibble())
      mutate(curves,
             split = unique(.x$split),
             task = unique(.x$task),
             fold = unique(.x$fold),
             model_label = unique(.x$model_label))
    }) %>%
    ungroup()
}

if (nrow(time_brier_dt) == 0) {
  message("未在 metrics_new.json 中读到时间依赖 Brier，改为直接基于 predictions.csv 计算……")
  time_brier_dt <- build_time_brier_from_predictions(pred_dt)
}

plot_time_brier <- function(split_name, task_name) {
  df <- time_brier_dt %>% filter(split == split_name, task == task_name, !is.na(brier))
  if (nrow(df) == 0) return(NULL)
  ggplot(df, aes(x = time, y = brier, color = model_label, group = interaction(model_label, fold))) +
    geom_line(alpha = 0.7) +
    scale_color_brewer(palette = "Set1") +
    labs(
      title = glue("{task_name} - {toupper(split_name)} 集 时间依赖 Brier 曲线"),
      subtitle = "每条轨迹 = 1 个模型 × 1 个折，展示截止时间的 Brier 分数",
      x = "时间 (与 *_time 列一致)",
      y = "Time-dependent Brier",
      color = "轨迹提取器 + 风险预测器"
    ) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "bottom", legend.title = element_text(size = 10))
}

if (nrow(time_brier_dt) > 0) {
  for (task_name in tasks) {
    for (split_name in splits) {
      plt <- plot_time_brier(split_name, task_name)
      if (is.null(plt)) next
      outfile <- file.path(output_dir, glue("brier_{tolower(task_name)}_{split_name}.png"))
      ggsave(outfile, plt, width = 10, height = 6, dpi = 300)
    }
  }
  message(glue("Brier 曲线亦已输出到: {output_dir}"))
} else {
  message("未能生成时间依赖 Brier 曲线（可能缺少时间戳或概率）。")
}

ibs_dt <- time_brier_dt %>%
  filter(!is.na(brier)) %>%
  group_by(split, task, fold, model_label) %>%
  summarise(
    ibs = mean(brier, na.rm = TRUE),
    n_points = n(),
    .groups = "drop"
  )

if (nrow(ibs_dt) > 0) {
  ibs_path <- file.path(output_dir, "brier_ibs.csv")
  fwrite(ibs_dt, ibs_path)
  message(glue("积分 Brier (IBS) 已保存: {ibs_path}"))
}

# --- 阈值扫描：Sensitivity / Specificity / PPV / NPV / F1 vs 阈值 ---

compute_threshold_metrics <- function(df, thresholds = NULL) {
  thresholds <- resolve_thresholds(thresholds, DEFAULT_THRESHOLD_GRID, "THRESHOLD_GRID")
  df <- df %>% filter(!is.na(probability), !is.na(label))
  if (nrow(df) < 5 || n_distinct(df$label) < 2) return(tibble())
  map_dfr(thresholds, function(th) {
    pred <- as.integer(df$probability >= th)
    tp <- sum(pred == 1 & df$label == 1)
    fp <- sum(pred == 1 & df$label == 0)
    tn <- sum(pred == 0 & df$label == 0)
    fn <- sum(pred == 0 & df$label == 1)
    sensitivity <- if ((tp + fn) > 0) tp / (tp + fn) else NA_real_
    specificity <- if ((tn + fp) > 0) tn / (tn + fp) else NA_real_
    ppv <- if ((tp + fp) > 0) tp / (tp + fp) else NA_real_
    npv <- if ((tn + fn) > 0) tn / (tn + fn) else NA_real_
    # 当 PPV 或 Sensitivity 缺失或分母为 0 时，F1 亦应为 NA，避免逻辑判断触发错误
    valid_f1 <- !is.na(ppv) && !is.na(sensitivity) && (ppv + sensitivity) > 0
    f1 <- if (valid_f1) 2 * ppv * sensitivity / (ppv + sensitivity) else NA_real_
    tibble(
      threshold = th,
      sensitivity = sensitivity,
      specificity = specificity,
      ppv = ppv,
      npv = npv,
      f1 = f1
    )
  })
}

threshold_metric_dt <- pred_dt %>%
  filter(split %in% c("val", "test")) %>%
  group_by(split, task, fold, model_label) %>%
  group_modify(~{
    curves <- compute_threshold_metrics(.x)
    if (nrow(curves) == 0) return(tibble())
    mutate(curves,
           split = unique(.x$split),
           task = unique(.x$task),
           fold = unique(.x$fold),
           model_label = unique(.x$model_label))
  }) %>%
  ungroup() %>%
  tidyr::pivot_longer(
    cols = c("sensitivity", "specificity", "ppv", "npv", "f1"),
    names_to = "metric",
    values_to = "value"
  )

metric_labels <- c(
  sensitivity = "Sensitivity (Recall)",
  specificity = "Specificity",
  ppv = "Positive Predictive Value",
  npv = "Negative Predictive Value",
  f1 = "F1 Score"
)

plot_threshold_metric <- function(split_name, task_name, metric_name) {
  df <- threshold_metric_dt %>% filter(split == split_name, task == task_name, metric == metric_name)
  if (nrow(df) == 0) return(NULL)
  ggplot(df, aes(x = threshold, y = value, color = model_label, group = interaction(model_label, fold))) +
    geom_line(alpha = 0.75) +
    scale_color_brewer(palette = "Set1") +
    coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
    labs(
      title = glue("{task_name} - {toupper(split_name)} 集 {metric_labels[[metric_name]]} vs 阈值"),
      subtitle = "每条曲线 = 1 个模型 × 1 个折",
      x = "Threshold",
      y = metric_labels[[metric_name]],
      color = "轨迹提取器 + 风险预测器"
    ) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "bottom", legend.title = element_text(size = 10))
}

if (nrow(threshold_metric_dt) > 0) {
  threshold_metrics_path <- file.path(output_dir, "threshold_metrics.csv")
  fwrite(threshold_metric_dt, threshold_metrics_path)
  for (task_name in tasks) {
    for (split_name in splits) {
      for (metric_name in names(metric_labels)) {
        plt <- plot_threshold_metric(split_name, task_name, metric_name)
        if (is.null(plt)) next
        outfile <- file.path(output_dir, glue("threshold_{metric_name}_{tolower(task_name)}_{split_name}.png"))
        ggsave(outfile, plt, width = 10, height = 6, dpi = 300)
      }
    }
  }
  message(glue("阈值扫描曲线与表格已输出，CSV: {threshold_metrics_path}"))
} else {
  message("阈值扫描数据不足，未生成 Sensitivity/Specificity/PPV/NPV/F1 vs 阈值 曲线。")
}

# --- MCC / Balanced Accuracy 分布（小提琴 + 箱线图） ---

aggregate_path <- file.path(results_dir, "metrics_aggregate_new.csv")
if (!file.exists(aggregate_path)) {
  message("未找到 metrics_aggregate_new.csv，无法绘制 MCC / Balanced Accuracy 分布图。")
} else {
  agg_dt <- fread(aggregate_path, encoding = "UTF-8") %>%
    mutate(
      split = tolower(split),
      task = as.character(task)
    )

  if (!all(c("Balanced_Accuracy", "MCC") %in% names(agg_dt))) {
    message("汇总表中缺少 Balanced_Accuracy 或 MCC 列，跳过分布图绘制。")
  } else {
    has_traj_cols <- all(c("trajectory", "risk") %in% names(agg_dt))
    has_model_label <- "model_label" %in% names(agg_dt)
    agg_dt <- agg_dt %>%
      filter(split %in% c("val", "test")) %>%
      mutate(
        model_label = if (has_traj_cols) {
          glue("{trajectory} + {risk}")
        } else if (has_model_label) {
          as.character(model_label)
        } else {
          paste0("model_", row_number())
        }
      )

    metric_long <- agg_dt %>%
      select(split, task, fold, model_label, Balanced_Accuracy, MCC) %>%
      tidyr::pivot_longer(
        cols = c(Balanced_Accuracy, MCC),
        names_to = "metric",
        values_to = "value"
      ) %>%
      filter(!is.na(value)) %>%
      mutate(model_label = forcats::fct_reorder(model_label, value, .fun = median, .na_rm = TRUE))

    metric_labels <- c(
      Balanced_Accuracy = "Balanced Accuracy",
      MCC = "Matthews Correlation Coefficient"
    )

    plot_metric_distribution <- function(metric_name) {
      df <- metric_long %>% filter(metric == metric_name)
      if (nrow(df) == 0) return(NULL)

      value_min <- min(df$value, na.rm = TRUE)
      value_max <- max(df$value, na.rm = TRUE)
      lower_bound <- if (metric_name == "MCC") min(-0.25, value_min - 0.05) else max(0, value_min - 0.05)
      upper_bound <- if (metric_name == "MCC") min(1, value_max + 0.05) else min(1.05, value_max + 0.05)

      ggplot(df, aes(x = value, y = model_label)) +
        geom_violin(
          fill = "#b3cde3",
          color = "#90a4ae",
          alpha = 0.7,
          scale = "width",
          trim = FALSE
        ) +
        geom_boxplot(
          width = 0.2,
          fill = "white",
          color = "#424242",
          outlier.shape = 21,
          outlier.fill = "#757575",
          outlier.alpha = 0.5
        ) +
        facet_grid(split ~ task, scales = "free_y", space = "free_y") +
        coord_cartesian(xlim = c(lower_bound, upper_bound)) +
        labs(
          title = glue("{metric_labels[[metric_name]]} 分布 (按模型 × 折)"),
          subtitle = "浅蓝小提琴展示密度，白色箱线突出中位数与 IQR",
          x = metric_labels[[metric_name]],
          y = "轨迹提取器 + 风险预测器"
        ) +
        theme_minimal(base_size = 12) +
        theme(
          legend.position = "none",
          strip.text = element_text(face = "bold"),
          axis.text.y = element_text(size = 9)
        )
    }

    for (metric_name in names(metric_labels)) {
      plt <- plot_metric_distribution(metric_name)
      if (is.null(plt)) next
      outfile <- file.path(output_dir, glue("distribution_{tolower(metric_name)}.png"))
      ggsave(outfile, plt, width = 12, height = 8, dpi = 300)
    }

    dist_csv <- file.path(output_dir, "distribution_mcc_balanced_accuracy.csv")
    fwrite(metric_long, dist_csv)
    message(glue("MCC / Balanced Accuracy 分布图与原始数据已输出，CSV: {dist_csv}"))
  }
}

# --- Decision Curve Analysis (Net Benefit) ---

compute_dca_curve <- function(df, thresholds = NULL) {
  thresholds <- resolve_thresholds(thresholds, DEFAULT_DCA_THRESHOLDS, "DCA_THRESHOLDS")
  df <- df %>% filter(!is.na(probability), !is.na(label))
  if (nrow(df) == 0 || n_distinct(df$label) < 2) return(tibble())
  N <- nrow(df)
  map_dfr(thresholds, function(th) {
    if (th <= 0 || th >= 1) return(tibble())
    pred <- as.integer(df$probability >= th)
    tp <- sum(pred == 1 & df$label == 1)
    fp <- sum(pred == 1 & df$label == 0)
    net_benefit <- (tp / N) - (fp / N) * (th / (1 - th))
    tibble(threshold = th, net_benefit = net_benefit)
  })
}

compute_dca_strategies <- function(labels, thresholds = NULL) {
  thresholds <- resolve_thresholds(thresholds, DEFAULT_DCA_THRESHOLDS, "DCA_THRESHOLDS")
  labels <- labels[!is.na(labels)]
  if (length(labels) == 0) return(tibble())
  prevalence <- mean(labels == 1)
  map_dfr(thresholds, function(th) {
    if (th <= 0 || th >= 1) return(tibble())
    treat_all <- prevalence - (1 - prevalence) * (th / (1 - th))
    tibble(
      threshold = th,
      model_label = c("Treat All", "Treat None"),
      net_benefit = c(treat_all, 0),
      curve_type = "strategy"
    )
  })
}

dca_model_dt <- pred_dt %>%
  filter(split %in% c("val", "test")) %>%
  group_by(split, task, fold, model_label) %>%
  group_modify(~{
    curves <- compute_dca_curve(.x)
    if (nrow(curves) == 0) return(tibble())
    mutate(curves,
           split = unique(.x$split),
           task = unique(.x$task),
           fold = unique(.x$fold),
           model_label = unique(.x$model_label),
           curve_type = "model")
  }) %>%
  ungroup()

dca_strategy_dt <- pred_dt %>%
  filter(split %in% c("val", "test")) %>%
  group_by(split, task, fold) %>%
  group_modify(~{
    curves <- compute_dca_strategies(.x$label)
    if (nrow(curves) == 0) return(tibble())
    mutate(curves,
           split = unique(.x$split),
           task = unique(.x$task),
           fold = unique(.x$fold))
  }) %>%
  ungroup()

if (nrow(dca_strategy_dt) > 0) {
  dca_strategy_dt <- dca_strategy_dt %>% mutate(model_label = factor(model_label))
}

dca_dt <- bind_rows(dca_model_dt, dca_strategy_dt)

generate_color_map <- function(labels) {
  labels <- unique(as.character(labels))
  strategy_labels <- c("Treat All", "Treat None")
  model_labels <- labels[!labels %in% strategy_labels]
  base_palette <- RColorBrewer::brewer.pal(9, "Set1")
  model_colors <- setNames(rep(base_palette, length.out = length(model_labels)), model_labels)
  extras <- c("Treat All" = "#000000", "Treat None" = "#7f7f7f")
  extras <- extras[names(extras) %in% labels]
  c(model_colors, extras)
}

plot_dca <- function(split_name, task_name) {
  df <- dca_dt %>% filter(split == split_name, task == task_name)
  if (nrow(df) == 0) return(NULL)
  color_map <- generate_color_map(df$model_label)
  dca_range <- range(resolve_thresholds(NULL, DEFAULT_DCA_THRESHOLDS, "DCA_THRESHOLDS"))
  ggplot(df, aes(x = threshold, y = net_benefit, color = model_label, linetype = curve_type,
                 group = interaction(model_label, fold))) +
    geom_line(alpha = 0.8, linewidth = 0.7) +
    scale_color_manual(values = color_map) +
    scale_linetype_manual(values = c(model = "solid", strategy = "dashed"), guide = "none") +
    coord_cartesian(xlim = dca_range) +
    labs(
      title = glue("{task_name} - {toupper(split_name)} 集 Decision Curve (Net Benefit)"),
      subtitle = "每条曲线 = 1 个模型 × 1 个折；虚线为 Treat All / Treat None",
      x = "Threshold Probability",
      y = "Net Benefit",
      color = "轨迹提取器 + 风险预测器"
    ) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "bottom", legend.title = element_text(size = 10))
}

if (nrow(dca_dt) > 0) {
  dca_csv <- file.path(output_dir, "decision_curve_data.csv")
  fwrite(dca_dt, dca_csv)
  for (task_name in tasks) {
    for (split_name in splits) {
      plt <- plot_dca(split_name, task_name)
      if (is.null(plt)) next
      outfile <- file.path(output_dir, glue("decision_curve_{tolower(task_name)}_{split_name}.png"))
      ggsave(outfile, plt, width = 10, height = 6, dpi = 300)
    }
  }
  message(glue("Decision Curve Analysis 图与数据已输出，CSV: {dca_csv}"))
} else {
  message("未能生成 Decision Curve Analysis 曲线（可能是样本过少或标签单一）。")
}

# --- Net Reclassification / IDI 热力图 ---

compute_pairwise_nri_idi <- function(df) {
  required_cols <- c("subject_id", "model_label", "probability", "label")
  if (!all(required_cols %in% names(df))) {
    warning("predictions.csv 缺少 subject_id 列，无法计算 NRI/IDI")
    return(tibble())
  }
  wide_df <- df %>%
    select(subject_id, label, model_label, probability) %>%
    distinct() %>%
    tidyr::pivot_wider(names_from = model_label, values_from = probability)
  models <- setdiff(names(wide_df), c("subject_id", "label"))
  if (length(models) < 2) return(tibble())
  combos <- expand.grid(
    base_model = models,
    compare_model = models,
    stringsAsFactors = FALSE
  ) %>%
    filter(base_model != compare_model)
  if (nrow(combos) == 0) return(tibble())
  purrr::pmap_dfr(combos, function(base_model, compare_model) {
    pa <- wide_df[[base_model]]
    pb <- wide_df[[compare_model]]
    labels <- wide_df$label
    valid <- !is.na(pa) & !is.na(pb) & !is.na(labels)
    if (sum(valid) < 5 || length(unique(labels[valid])) < 2) return(tibble())
    pa <- pa[valid]
    pb <- pb[valid]
    labels <- labels[valid]
    diff <- pb - pa
    event_mask <- labels == 1
    non_event_mask <- labels == 0
    if (!any(event_mask) || !any(non_event_mask)) return(tibble())
    event_nri <- mean(diff[event_mask] > 0) - mean(diff[event_mask] < 0)
    non_event_nri <- mean(diff[non_event_mask] < 0) - mean(diff[non_event_mask] > 0)
    disc_base <- mean(pa[event_mask]) - mean(pa[non_event_mask])
    disc_comp <- mean(pb[event_mask]) - mean(pb[non_event_mask])
    tibble(
      base_model = base_model,
      compare_model = compare_model,
      nri = event_nri + non_event_nri,
      idi = disc_comp - disc_base
    )
  })
}

nri_idi_fold_dt <- pred_dt %>%
  filter(split %in% c("val", "test")) %>%
  group_by(split, task, fold) %>%
  group_modify(~{
    pairwise <- compute_pairwise_nri_idi(.x)
    if (nrow(pairwise) == 0) return(tibble())
    mutate(pairwise,
           split = unique(.x$split),
           task = unique(.x$task),
           fold = unique(.x$fold))
  }) %>%
  ungroup()

if (nrow(nri_idi_fold_dt) > 0) {
  nri_idi_dt <- nri_idi_fold_dt %>%
    group_by(split, task, base_model, compare_model) %>%
    summarise(
      nri = mean(nri, na.rm = TRUE),
      idi = mean(idi, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    tidyr::pivot_longer(cols = c("nri", "idi"), names_to = "metric", values_to = "value")

  nri_idi_csv <- file.path(output_dir, "nri_idi_heatmap_data.csv")
  fwrite(nri_idi_dt, nri_idi_csv)

  plot_nri_heatmap <- function(split_name, task_name, metric_name) {
    df <- nri_idi_dt %>%
      filter(split == split_name, task == task_name, metric == metric_name)
    if (nrow(df) == 0) return(NULL)
    level_candidates <- if (exists("model_levels")) model_levels else sort(unique(pred_dt$model_label))
    df <- df %>%
      mutate(
        base_model = factor(base_model, levels = level_candidates),
        compare_model = factor(compare_model, levels = level_candidates)
      )
    max_abs <- max(abs(df$value), na.rm = TRUE)
    if (!is.finite(max_abs) || max_abs == 0) {
      max_abs <- 1
    }
    ggplot(df, aes(x = compare_model, y = base_model, fill = value)) +
      geom_tile(color = "white", linewidth = 0.2) +
      geom_text(aes(label = sprintf("%.2f", value)), size = 2.4, na.rm = TRUE) +
      scale_fill_gradient2(
        limits = c(-max_abs, max_abs),
        low = "#2166ac",
        mid = "white",
        high = "#b2182b",
        midpoint = 0,
        name = toupper(metric_name)
      ) +
      labs(
        title = glue("{task_name} - {toupper(split_name)} 集 {toupper(metric_name)} 热力图"),
        subtitle = "行 = 基准模型，列 = 对比模型 (值 > 0 表示列模型优于行模型)",
        x = "对比模型",
        y = "基准模型"
      ) +
      theme_minimal(base_size = 11) +
      theme(
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "right"
      )
  }

  for (task_name in tasks) {
    for (split_name in splits) {
      for (metric_name in c("nri", "idi")) {
        plt <- plot_nri_heatmap(split_name, task_name, metric_name)
        if (is.null(plt)) next
        outfile <- file.path(output_dir, glue("heatmap_{metric_name}_{tolower(task_name)}_{split_name}.png"))
        ggsave(outfile, plt, width = 10, height = 8, dpi = 300)
      }
    }
  }
  message(glue("NRI/IDI 热力图与数据已输出，CSV: {nri_idi_csv}"))
} else {
  message("缺少 subject_id 或有效标签，跳过 NRI/IDI 热力图。")
}

# --- BayesianRiskAggregator 置信区间带 ---

has_ci_cols <- all(c("prob_lower", "prob_upper") %in% names(pred_dt))
ci_band_bins <- 100

if (!has_ci_cols) {
  message("predictions.csv 未包含 prob_lower/prob_upper，跳过 BayesianRiskAggregator 置信区间图。")
} else {
  ci_dt <- pred_dt %>%
    filter(split %in% c("val", "test"), risk == "bayesian_laplace") %>%
    filter(!is.na(prob_lower), !is.na(prob_upper))

  if (nrow(ci_dt) == 0) {
    message("BayesianRiskAggregator 未输出置信区间，跳过置信区间带绘图。")
  } else {
    ci_summary <- ci_dt %>%
      group_by(task, split, trajectory, fold) %>%
      arrange(desc(probability), .by_group = TRUE) %>%
      mutate(
        rank = if (n() <= 1) 1 else (row_number() - 1) / (n() - 1),
        bin = cut(rank,
                  breaks = seq(0, 1, length.out = ci_band_bins + 1),
                  include.lowest = TRUE,
                  labels = FALSE)
      ) %>%
      ungroup() %>%
      group_by(task, split, trajectory, bin) %>%
      summarise(
        rank = mean(rank, na.rm = TRUE),
        prob = mean(probability, na.rm = TRUE),
        lower = mean(prob_lower, na.rm = TRUE),
        upper = mean(prob_upper, na.rm = TRUE),
        .groups = "drop"
      ) %>%
      filter(!is.na(rank), !is.na(lower), !is.na(upper))

    plot_ci_band <- function(split_name, task_name) {
      df <- ci_summary %>% filter(split == split_name, task == task_name)
      if (nrow(df) == 0) return(NULL)
      ggplot(df, aes(x = rank, y = prob, color = trajectory, fill = trajectory)) +
        geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.15, color = NA) +
        geom_line(linewidth = 0.9) +
        coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
        labs(
          title = glue("{task_name} - {toupper(split_name)} 集 BayesianRiskAggregator 置信区间"),
          subtitle = "横轴为按概率排序后的累计占比，阴影区表示上下置信带",
          x = "累计样本占比 (0-1)",
          y = "预测概率",
          color = "轨迹提取器",
          fill = "轨迹提取器"
        ) +
        theme_minimal(base_size = 12) +
        theme(legend.position = "bottom", legend.title = element_text(size = 10))
    }

    ci_csv <- file.path(output_dir, "bayesian_ci_summary.csv")
    fwrite(ci_summary, ci_csv)
    for (task_name in tasks) {
      for (split_name in splits) {
        plt <- plot_ci_band(split_name, task_name)
        if (is.null(plt)) next
        outfile <- file.path(output_dir, glue("bayesian_ci_{tolower(task_name)}_{split_name}.png"))
        ggsave(outfile, plt, width = 10, height = 6, dpi = 300)
      }
    }
    message(glue("BayesianRiskAggregator 置信区间图与数据已输出，CSV: {ci_csv}"))
  }
}

# --- 任务 (T2 vs T3) 雷达图 ---

radar_metrics <- c("AUC", "Sensitivity", "Specificity", "PPV", "NPV", "F1_Score", "Balanced_Accuracy")

if (!file.exists(aggregate_path)) {
  message("未找到 metrics_aggregate_new.csv，无法绘制任务雷达图。")
} else {
  mean_or_na <- function(x) {
    if (all(is.na(x))) {
      return(NA_real_)
    }
    mean(x, na.rm = TRUE)
  }

  radar_dt <- fread(aggregate_path, encoding = "UTF-8") %>%
    mutate(split = tolower(split), task = as.character(task))

  available_metrics <- radar_metrics[radar_metrics %in% names(radar_dt)]
  if (length(available_metrics) < 3) {
    message("汇总表中可用于雷达图的指标不足，跳过任务对比图。")
  } else {
    radar_summary <- radar_dt %>%
      filter(split %in% c("val", "test")) %>%
      group_by(split, task) %>%
      summarise(across(all_of(available_metrics), mean_or_na), .groups = "drop") %>%
      filter(if_any(all_of(available_metrics), ~ !is.na(.x)))

    radar_long <- radar_summary %>%
      tidyr::pivot_longer(cols = all_of(available_metrics), names_to = "metric", values_to = "value") %>%
      filter(!is.na(value)) %>%
      mutate(metric = factor(metric, levels = available_metrics))

    plot_radar <- function(split_name) {
      df <- radar_long %>% filter(split == split_name)
      if (nrow(df) == 0) return(NULL)

      df_ordered <- df %>% arrange(task, metric)
      df_closed <- df_ordered %>%
        group_by(task) %>%
        arrange(metric, .by_group = TRUE) %>%
        bind_rows(slice_head(., n = 1)) %>%
        ungroup()

      ggplot() +
        geom_polygon(
          data = df_closed,
          aes(x = metric, y = value, group = task, color = task, fill = task),
          alpha = 0.15,
          linewidth = 0.8
        ) +
        geom_line(
          data = df_ordered,
          aes(x = metric, y = value, group = task, color = task),
          linewidth = 0.9
        ) +
        geom_point(
          data = df_ordered,
          aes(x = metric, y = value, color = task),
          size = 2.4
        ) +
        coord_polar(start = -pi / 2, clip = "off") +
        scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.2)) +
        labs(
          title = glue("{toupper(split_name)} 集 T2 vs T3 任务雷达图"),
          subtitle = "多指标平均表现 (取各模型在该 split 的均值)",
          x = NULL,
          y = NULL,
          color = "任务",
          fill = "任务"
        ) +
        theme_minimal(base_size = 12) +
        theme(
          legend.position = "bottom",
          panel.grid.major = element_line(color = "#d9d9d9"),
          axis.text.x = element_text(face = "bold")
        )
    }

    radar_csv <- file.path(output_dir, "task_radar_summary.csv")
    fwrite(radar_summary, radar_csv)
    for (split_name in splits) {
      plt <- plot_radar(split_name)
      if (is.null(plt)) next
      outfile <- file.path(output_dir, glue("task_radar_{split_name}.png"))
      ggsave(outfile, plt, width = 8, height = 8, dpi = 300)
    }
    message(glue("任务雷达图与数据已输出，CSV: {radar_csv}"))
  }
}
