# ===========================================================================
# File Name: 23_run_misty_c2l_analysis.R
# Author: Kuroneko
# Date: 2026-02-04
# Version: V2.1 (Final Fixed Edition)
#
# Description:
#     [MISTy 空间依赖性建模 - 最终修复版]
#
#     【背景与科学意义】
#     在前面的步骤中，Cell2location 告诉了我们“每个点里有多少细胞 (Abundance)”。
#     但细胞不是孤岛，它们受环境影响。MISTy (Multiview Intercellular SpaTial modeling framework)
#     是一个可解释的机器学习框架，旨在回答核心的生物学问题：
#     1. 细胞的分布主要由什么决定？是它自己的内在特性 (Intra)，还是邻居的影响 (Para)？
#     2. 共定位 (Colocalization): T细胞是否总是伴随着 B细胞出现？
#     3. 排斥作用 (Segregation): 肿瘤细胞是否在排斥免疫细胞？
#
#     【逻辑流程】
#     1. 数据准备: 加载去卷积得到的细胞丰度矩阵 + 空间坐标。
#     2. 数据清洗: 修复非法的列名 (如空格、斜杠、特殊符号)，这是机器学习模型的硬性要求。
#     3. 视图构建 (View Construction):
#        - Intra-view: 自身的细胞组成。
#        - Para-view: 邻居的细胞组成 (基于动态计算的距离阈值)。
#     4. 模型训练: 为每种细胞类型训练 Meta-model (随机森林 + 岭回归)。
#     5. 结果可视化: 生成重要性评分图、互作热图、网络图。
#
# Input Files:
#     - Abundance: F:/ST/code/results/deconvolution/cell_abundance.csv
#     - Spatial: F:/ST/code/data/V1_Human_Lymph_Node/
#
# Output Files (in F:/ST/code/results/misty/):
#     - plots/: 包含改进度图、互作热图、网络图。
#     - misty_results.rds: 最终结果对象。
# ===========================================================================

# -------------------------------------------------------------------------
# [Module 1] 环境初始化与依赖检查
# -------------------------------------------------------------------------
# 清空内存，防止干扰
rm(list = ls())
gc()

message("[Module 1] Initializing Environment...")

# 定义所需的包列表
required_pkgs <- c("mistyR", "future", "Seurat", "tidyverse", "distances", "ggplot2", "ranger", "ridge")

# 检查并安装缺失的包
for (pkg in required_pkgs) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message(paste0("Installing missing package: ", pkg))
    if (pkg == "mistyR") {
      options(BioC_mirror = "https://mirrors.westlake.edu.cn/bioconductor")
      BiocManager::install("mistyR")
    } else {
      install.packages(pkg)
    }
  }
}

library(mistyR)
library(future)
library(Seurat)
library(tidyverse)
library(distances)
library(ggplot2)
library(ranger) # 确保加载
library(ridge)  # 确保加载

# 设置并行计算 (Windows 推荐 multisession)
# MISTy 训练非常耗时，开启多核加速
# 注意：如果内存爆了，可以把 workers 调小到 4
plan(multisession, workers = 8)
options(future.globals.maxSize = 8000 * 1024^2)

# 路径设置
BASE_DIR <- "F:/ST/code/"
ABUNDANCE_FILE <- file.path(BASE_DIR, "results/deconvolution/cell_abundance.csv")
SPATIAL_DIR <- file.path(BASE_DIR, "data/V1_Human_Lymph_Node")
OUT_DIR <- file.path(BASE_DIR, "results/misty")

# 创建输出目录
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
plot_path <- file.path(OUT_DIR, "plots")
dir.create(plot_path, showWarnings = FALSE)

message("    Environment Ready. Parallel workers: 8")

# -------------------------------------------------------------------------
# [Module 2] 数据加载与清洗 (CRITICAL FIXES HERE)
# -------------------------------------------------------------------------
message("[Module 2] Loading and Sanitizing Data...")

# 1. 加载 Cell2location 的去卷积结果
if (!file.exists(ABUNDANCE_FILE)) {
  stop("Error: cell_abundance.csv not found! Please run Step 0 (C2L) first.")
}

# check.names = FALSE 先读进来，后面我们自己处理
abundance <- read.csv(ABUNDANCE_FILE, row.names = 1, check.names = FALSE)
message(paste0("    Loaded Abundance Matrix: ", nrow(abundance), " spots x ", ncol(abundance), " cell types"))

# 2. [关键修复] 清洗列名
# ranger/ridge 等机器学习包不支持列名中包含 空格、斜杠、加号、减号 等特殊字符。
# make.names 会自动把这些字符替换为点号 (.)
message("    Sanitizing column names for machine learning compatibility...")
original_names <- colnames(abundance)
colnames(abundance) <- make.names(colnames(abundance))

# 打印前后对比，让你确认
print(data.frame(Original = head(original_names, 3), Sanitized = head(colnames(abundance), 3)))

# 3. 加载空间坐标
message("    Loading Spatial Coordinates...")
visium_obj <- Load10X_Spatial(SPATIAL_DIR)
coords <- GetTissueCoordinates(visium_obj, scale = NULL)

# 统一重命名坐标列为 MISTy 要求的格式
geometry <- coords[, 1:2]
colnames(geometry) <- c("row", "col") 

# 4. 数据对齐 (Intersection)
common_spots <- intersect(rownames(abundance), rownames(geometry))

if (length(common_spots) == 0) {
  stop("Error: No common spots found! Check barcodes format.")
}

abundance <- abundance[common_spots, ]
geometry <- geometry[common_spots, ]

message(paste0("    Aligned Data: ", length(common_spots), " spots."))

# -------------------------------------------------------------------------
# [Module 3] 构建 MISTy Views (CRITICAL FIXES HERE)
# -------------------------------------------------------------------------
message("[Module 3] Constructing Views...")

# 1. Intra-view (内在视图)
intra_view <- create_initial_view(abundance)

# 2. 自动计算 Para-view 半径 (l)
geom_dist <- as.matrix(distances(geometry))
dist_nn <- apply(geom_dist, 1, function(x) sort(x)[2]) # 第2小距离(最近邻)
paraview_radius <- ceiling(mean(dist_nn + sd(dist_nn)))

message(paste0("    -> Calculated dynamic radius (l): ", paraview_radius))

# 3. 添加 Para-view (邻居视图)
# [修复说明] add_paraview 不支持 name 参数，必须先生成再改名
message("    Generating Para-view (1x radius)...")
misty_views <- intra_view %>%
  add_paraview(geometry, l = paraview_radius, family = "gaussian")

# [关键修复] 手动重命名最后一个视图
names(misty_views)[length(misty_views)] <- "para_1x"

# 4. (可选) 添加 2倍半径视图
message("    Generating Para-view (2x radius)...")
misty_views <- misty_views %>%
  add_paraview(geometry, l = paraview_radius * 2, family = "gaussian")

# 手动重命名
names(misty_views)[length(misty_views)] <- "para_2x"

# 打印检查最终的视图结构
message("    Final Views created:")
print(names(misty_views))
# 预期输出: "intraview", "misty.uniqueid", "para_1x", "para_2x"

# -------------------------------------------------------------------------
# [Module 4] 运行 MISTy 模型
# -------------------------------------------------------------------------
message("[Module 4] Running MISTy (Training Random Forest)...")
message("    This may take 5-10 minutes depending on your CPU...")

# 结果临时目录
misty_out_folder <- file.path(OUT_DIR, "misty_run_c2l")
# 如果目录存在则清理，防止旧文件干扰
if(dir.exists(misty_out_folder)) unlink(misty_out_folder, recursive = TRUE)

# 运行训练
# 如果之前报 ranger 或 ridge 错误，Module 1 的安装步骤已经解决了
run_misty(misty_views, misty_out_folder)

# 收集结果
misty_results <- collect_results(misty_out_folder)

message("    -> Model training and collection finished.")

# -------------------------------------------------------------------------
# [Module 5 Final] 完整可视化与结果导出
# -------------------------------------------------------------------------
message(">>>>>>>>>> [Module 5] Visualization START <<<<<<<<<<")

library(dplyr)
library(ggplot2)
library(mistyR) # 确保加载

# =========================================================
# 1. 智能视图名称检测
# =========================================================
# 自动从结果中提取 View 的名字 (例如 para.142)
available_views <- unique(misty_results$importances.aggregated$view)
message("    Views found: ", paste(available_views, collapse = ", "))

# 筛选 para 视图
para_views <- grep("para", available_views, value = TRUE)
view_1x_name <- para_views[1]
view_2x_name <- if(length(para_views) > 1) para_views[2] else NULL

# =========================================================
# 2. 全局数据清洗 (去除 Cell2location 长前缀)
# =========================================================
message("    Cleaning cell type names...")

# 定义清洗逻辑
clean_names <- function(x) {
  x <- gsub("q05cell_abundance_w_sf_means_per_cluster_mu_fg_", "", x)
  x <- gsub("means_cell_abundance_w_sf_means_per_cluster_mu_fg_", "", x)
  x <- gsub("_", " ", x) # 将下划线替换为空格，更美观
  return(x)
}

# 清洗重要性表 (用于热图)
misty_results$importances.aggregated <- misty_results$importances.aggregated %>%
  mutate(
    Predictor = clean_names(Predictor),
    Target = clean_names(Target)
  )

# 清洗改进度表 (用于 Gain R2 图)
misty_results$improvements <- misty_results$improvements %>%
  mutate(target = clean_names(target))

# =========================================================
# 3. 绘制所有图表
# =========================================================

# --- 图 1: 改进度统计 (Gain R2) ---
# 既然名字已经清洗，这里的图表坐标轴会自动变短
pdf(file.path(plot_path, "01_Spatial_Dependence_GainR2.pdf"), width = 12, height = 8)
p_r2 <- misty_results %>% 
  plot_improvement_stats("gain.R2") + 
  ggtitle("Spatial Dependence (Gain in R2)") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 10)) 
print(p_r2)
dev.off()
message("    -> 01_Spatial_Dependence_GainR2.pdf saved.")

# --- 图 2: 视图贡献度 (View Contributions) ---
pdf(file.path(plot_path, "02_View_Contributions.pdf"), width = 12, height = 8)
p_view <- misty_results %>% plot_view_contributions() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 10))
print(p_view)
dev.off()
message("    -> 02_View_Contributions.pdf saved.")

# --- 图 3 & 4: 空间共定位热图 (使用 ggplot2 手动绘制) ---
# 设定阈值 (你之前最高分是 5.49，这里设 1.0 可以过滤掉低分噪音)
HEATMAP_CUTOFF <- 1.0 

# 定义一个通用的手动绘图函数 (保证不出错)
draw_custom_heatmap <- function(data, view_name, cutoff) {
  # 1. 过滤数据
  plot_data <- data %>%
    filter(view == view_name, Importance >= cutoff)
  
  # 2. 检查是否有数据
  if (nrow(plot_data) == 0) {
    message(paste0("    [Warning] No interactions found > ", cutoff, " for ", view_name))
    return(NULL)
  }
  
  # 3. 绘图
  p <- ggplot(plot_data, aes(x = Target, y = Predictor, fill = Importance)) +
    geom_tile(color = "white") +
    scale_fill_gradient(low = "#fee0d2", high = "#de2d26") + # 红色系热图
    theme_minimal() +
    labs(title = paste0("Colocalization (", view_name, ")"),
         subtitle = paste0("Importance Cutoff: ", cutoff)) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
          axis.text.y = element_text(size = 10),
          panel.grid = element_blank())
  return(p)
}

# 绘制 1x 热图
if (!is.null(view_1x_name)) {
  pdf(file.path(plot_path, "03_Colocalization_Heatmap_1x.pdf"), width = 14, height = 12)
  p3 <- draw_custom_heatmap(misty_results$importances.aggregated, view_1x_name, HEATMAP_CUTOFF)
  if (!is.null(p3)) print(p3)
  dev.off()
  message("    -> 03_Colocalization_Heatmap_1x.pdf saved.")
}

# 绘制 2x 热图
if (!is.null(view_2x_name)) {
  pdf(file.path(plot_path, "03_Colocalization_Heatmap_2x.pdf"), width = 14, height = 12)
  p4 <- draw_custom_heatmap(misty_results$importances.aggregated, view_2x_name, HEATMAP_CUTOFF)
  if (!is.null(p4)) print(p4)
  dev.off()
  message("    -> 03_Colocalization_Heatmap_2x.pdf saved.")
}

# =========================================================
# 4. 导出统一的 CSV 文件
# =========================================================
# 覆盖之前的旧文件，不再生成 DEBUG 文件
csv_path <- file.path(OUT_DIR, "misty_colocalization_scores.csv")

# 筛选需要的视图和列
target_views <- c(view_1x_name, view_2x_name)
target_views <- target_views[!is.null(target_views)]

final_table <- misty_results$importances.aggregated %>%
  filter(view %in% target_views) %>%
  arrange(desc(Importance)) %>%
  select(Target, Predictor, Importance, view, nsamples)

write.csv(final_table, csv_path, row.names = FALSE)
message(paste0("    -> CSV updated: ", csv_path))

# 保存最终对象 (覆盖旧文件)
saveRDS(misty_results, file.path(OUT_DIR, "misty_c2l_results.rds"))

message(">>>>>>>>>> [Module 5] Visualization DONE <<<<<<<<<<")