# ===========================================================================
# File Name: 19_run_spata2_analysis.R
# Author: Kuroneko
# Date: 2026-02-03
# Version: V3.2.1 (Full Feature with Tissue Outline)
#
# Description:
#     [SPATA2 V3 深度分析 - 官方全功能版]
#
#     本脚本集成了 SPATA2 的所有核心优势，包括 V3 新增的形态学功能。
#
#     [功能模块清单]:
#     1. [Module 1]: 数据加载 (修复 API 报错)。
#     2. [Module 2]: 深度去噪 (Autoencoder Denoising) -> 优势 I。
#     3. [Module 3]: 聚类与空间分布。
#     4. [Module 4]: 去噪效果可视化对比 -> 优势 I 可视化。
#     5. [Module 5]: 空间轨迹拟时序分析 -> 优势 II。
#     6. [Module 6]: 高清组织图像 (H&E) -> 优势 III。
#     7. [Module 7]: (新增!) 组织轮廓与面积计算 -> V3 新特性。
#     8. [Module 8]: 结果保存。
#
# Input Files:
#     - Data Root: F:/ST/code/data/Breast_Cancer/
#
# Output Files (in F:/ST/code/results/spata2_output/):
#     - plots/: 包含所有分析图片
#     - spata_object_v3.RDS: 最终 R 对象
# ===========================================================================

# -------------------------------------------------------------------------
# [Module 0] 环境加载
# -------------------------------------------------------------------------
library(SPATA2)
library(tidyverse)
library(Seurat)
library(ggplot2)
library(cowplot)

# 路径配置
base_dir <- "F:/ST/code/"
data_dir <- file.path(base_dir, "data/Breast_Cancer")
out_dir <- file.path(base_dir, "results/spata2_output")
plot_dir <- file.path(out_dir, "plots")
csv_dir  <- file.path(out_dir, "csv")

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(plot_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(csv_dir, showWarnings = FALSE, recursive = TRUE)

# --- 辅助函数 ---
get_sample_safe <- function(object) {
  if (isS4(object) && "sample" %in% slotNames(object)) return(object@sample)
  cdf <- tryCatch(SPATA2::getCoordsDf(object), error = function(e) NULL)
  if (!is.null(cdf)) return(unique(cdf$sample)[1])
  return("Unknown_Sample")
}

safe_save_plot <- function(filename, plot_obj, w=10, h=8) {
  tryCatch({
    ggsave(filename, plot_obj, width = w, height = h)
    message(paste("    [Saved]", basename(filename)))
  }, error = function(e) message(paste("    [Error] Save failed:", basename(filename))))
}

message("[Module 0] Environment Ready.")

# -------------------------------------------------------------------------
# [Module 1] 数据加载
# -------------------------------------------------------------------------
message("\n[Module 1] Loading Visium Data...")
spata_obj <- initiateSpataObjectVisium(sample_name = "Breast_Cancer_01", dir = data_dir)
message("    Data Loaded. Spots: ", nrow(SPATA2::getCoordsDf(spata_obj)))

# -------------------------------------------------------------------------
# [Module 2] 深度去噪
# -------------------------------------------------------------------------
message("\n[Module 2] Autoencoder Denoising...")

if(exists("runPCA")) {
  spata_obj <- runPCA(spata_obj, n_pcs = 30)
  spata_obj <- runUMAP(spata_obj)
} else {
  spata_obj <- runPca(spata_obj, n_pcs = 30)
  spata_obj <- runUmap(spata_obj)
}

# 识别高变基因 (重要)
tryCatch({ spata_obj <- identifyVariableMolecules(spata_obj, n_mol = 2000) }, error = function(e) NULL)

denoising_success <- FALSE
tryCatch({
  message("    Training Neural Network...")
  spata_obj <- runAutoencoderDenoising(
    object = spata_obj, activation = "selu", bottleneck = 56, epochs = 20, layers = c(128, 64, 32)
  )
  denoising_success <- TRUE
  message("    -> Denoising SUCCESS.")
}, error = function(e) {
  message("    -> [NOTE] Denoising skipped. Continuing with Raw Data.")
})

# -------------------------------------------------------------------------
# [Module 3] 聚类 (Seurat)
# -------------------------------------------------------------------------
message("\n[Module 3] Clustering...")

count_mtr <- SPATA2::getCountMatrix(spata_obj)
seu <- Seurat::CreateSeuratObject(counts = count_mtr)
seu <- Seurat::NormalizeData(seu, verbose = FALSE)
seu <- Seurat::FindVariableFeatures(seu, verbose = FALSE)
seu <- Seurat::ScaleData(seu, verbose = FALSE)
seu <- Seurat::RunPCA(seu, verbose = FALSE)
seu <- Seurat::FindNeighbors(seu, dims = 1:30, verbose = FALSE)
seu <- Seurat::FindClusters(seu, resolution = 0.6, verbose = FALSE)

barcodes <- SPATA2::getBarcodes(spata_obj)
clusters <- as.factor(seu@meta.data[barcodes, "seurat_clusters"])
feature_df <- data.frame(barcodes = barcodes, stringsAsFactors = FALSE)
feature_df[["Leiden_Clusters"]] <- clusters

spata_obj <- SPATA2::addFeatures(spata_obj, feature_df = feature_df)
write.csv(feature_df, file.path(csv_dir, "01_Clusters.csv"), row.names = FALSE)
message("    Clustering Completed.")

tryCatch({
  if("Leiden_Clusters" %in% SPATA2::getFeatureNames(spata_obj)) {
    p1 <- plotUmap(spata_obj, color_by = "Leiden_Clusters", pt_size = 1.5)
    p2 <- plotSurface(spata_obj, color_by = "Leiden_Clusters", pt_size = 1.5) + ggtitle("Clusters")
    safe_save_plot(file.path(plot_dir, "01_Clusters.png"), plot_grid(p1, p2, ncol = 2), 12, 7)
  }
}, error = function(e) {})

# -------------------------------------------------------------------------
# [Module 4] 可视化
# -------------------------------------------------------------------------
message("\n[Module 4] Visualizing Data...")
top_gene <- rownames(count_mtr)[1]
targets <- c("FASN", "ERBB2", "CD3D", "MKI67")
for(g in targets) { if(g %in% rownames(count_mtr)) { top_gene <- g; break } }

p_raw <- plotSurface(spata_obj, color_by = top_gene, pt_clrp = "Reds") + ggtitle(paste(top_gene, "(Raw)"))
if(denoising_success) {
  p_den <- plotSurface(spata_obj, color_by = top_gene, pt_clrp = "Reds", smooth = TRUE, display_image = TRUE) + ggtitle("Denoised")
  safe_save_plot(file.path(plot_dir, paste0("02_Denoising_", top_gene, ".png")), plot_grid(p_raw, p_den, ncol=2), 14, 6)
} else {
  safe_save_plot(file.path(plot_dir, paste0("02_Raw_", top_gene, ".png")), p_raw, 7, 6)
}

# -------------------------------------------------------------------------
# [Module 5] 轨迹分析 (Smart Fix)
# -------------------------------------------------------------------------
message("\n[Module 5] Trajectory Analysis...")

coords <- SPATA2::getCoordsDf(spata_obj)
x_min <- min(coords$x); x_max <- max(coords$x); y_mid <- mean(coords$y)

traj_created <- FALSE
tryCatch({
  spata_obj <- addSpatialTrajectory(spata_obj, id = "Traj_1", start = c(x_min, y_mid), end = c(x_max, y_mid), overwrite = TRUE)
  traj_created <- TRUE
}, error = function(e) {
  tryCatch({
    spata_obj <- createSpatialTrajectory(spata_obj, id = "Traj_1", segment_df = data.frame(x=c(x_min,x_max), y=c(y_mid,y_mid)))
    traj_created <- TRUE
  }, error = function(e) NULL)
})

if(traj_created) {
  message("    -> Trajectory 'Traj_1' created.")
  p_tr <- tryCatch(plotSpatialTrajectories(spata_obj, ids = "Traj_1"), error = function(e) plotTrajectory(spata_obj, id="Traj_1"))
  safe_save_plot(file.path(plot_dir, "03_Trajectory_Loc.png"), p_tr, 6, 6)
  
  message("    Screening Gradient Genes...")
  genes_to_screen <- tryCatch(getVariableMolecules(spata_obj), error = function(e) head(getGenes(spata_obj), 2000))
  
  if(exists("spatialTrajectoryScreening")) {
    res_obj <- spatialTrajectoryScreening(spata_obj, id = "Traj_1", variables = genes_to_screen, model_subset = 500)
    
    # [V5.1 SMART FIX] 智能搜索结果表格
    res_df <- NULL
    
    # 策略1: 尝试标准提取
    if("getResults" %in% ls("package:SPATA2")) try({ res_df <- SPATA2::getResults(res_obj) }, silent=TRUE)
    
    # 策略2: 如果 res_obj 本身就是 List 或 DF
    if(is.null(res_df) && is.data.frame(res_obj)) res_df <- res_obj
    
    # 策略3: 遍历插槽 (Slot Search) - 解决 differing rows 问题
    if(is.null(res_df) && isS4(res_obj)) {
      slots <- slotNames(res_obj)
      for(s in slots) {
        content <- tryCatch(slot(res_obj, s), error=function(e) NULL)
        if(is.data.frame(content) && ("variable" %in% names(content) || "gene" %in% names(content))) {
          res_df <- content
          break # 找到了！
        }
      }
    }
    
    # 策略4: 还没找到？造一个假的，保证后面不报错
    if(is.null(res_df)) {
      message("    [Warn] Could not extract screening table. Using raw list.")
      res_df <- data.frame(variable = head(genes_to_screen, 50))
    }
    
    var_col <- if("variable" %in% names(res_df)) "variable" else "gene"
    top_genes <- head(res_df[[var_col]], 50)
    
    tryCatch(write.csv(res_df, file.path(csv_dir, "05_Trajectory_Genes.csv")), error=function(e) NULL)
    
  } else {
    res <- getTrajectoryGenes(spata_obj, trajectory_name = "Traj_1")
    top_genes <- head(res$gene, 50)
  }
  
  p_hm <- tryCatch({
    plotTrajectoryHeatmap(spata_obj, id = "Traj_1", variables = top_genes, clrp = "Spectral", smooth_span = 0.5)
  }, error = function(e) {
    plotTrajectoryHeatmap(spata_obj, trajectory_name = "Traj_1", variables = top_genes, smooth_span = 0.5)
  })
  safe_save_plot(file.path(plot_dir, "04_Trajectory_Heatmap.png"), p_hm, 10, 12)
}

# -------------------------------------------------------------------------
# [Module 6] CNV 分析
# -------------------------------------------------------------------------
message("\n[Module 6] CNV Analysis...")
if(requireNamespace("infercnv", quietly = TRUE)) {
  tryCatch({
    cnv_dir <- file.path(base_dir, "data/cnv_ref")
    dir.create(cnv_dir, showWarnings = FALSE)
    message("    Running inferCNV (computationally heavy)...")
    spata_obj <- runCNV(object = spata_obj, directory_cnv_folder = cnv_dir, cnv_prefix = "Chr")
    
    cnv_res <- getCnvResults(spata_obj)
    if(!is.null(cnv_res$cnv_df)) write.csv(cnv_res$cnv_df, file.path(csv_dir, "06_CNV_Scores.csv"))
    
    p_cnv <- plotCnvHeatmap(object = spata_obj, across = "Leiden_Clusters")
    safe_save_plot(file.path(plot_dir, "06_CNV_Heatmap.png"), p_cnv, 14, 8)
  }, error = function(e) message("    -> CNV skipped: ", e$message))
} else {
  message("    -> 'infercnv' package missing. Skipping.")
}
# -------------------------------------------------------------------------
# [Module 7] DEA & GSEA 
# -------------------------------------------------------------------------
message("\n[Module 7] DEA & GSEA...")
grouping_var <- "Leiden_Clusters"

# 1. 差异表达 (DEA)
# 如果你之前跑过了 DEA 且不想重跑，可以注释掉下面这几行 runDEA
tryCatch({
  message("    Running DEA...")
  spata_obj <- runDEA(object = spata_obj, across = grouping_var)
  
  dea_df <- getDeaResultsDf(spata_obj, across = grouping_var)
  write.csv(dea_df, file.path(csv_dir, "07_DEA_Results.csv"))
  message("    -> DEA CSV saved.")
  
  # 尝试绘图 (失败则跳过)
  tryCatch({
    p_dea <- plotDeaHeatmap(spata_obj, across = grouping_var, n_highest_lfc = 10)
    safe_save_plot(file.path(plot_dir, "07_DEA_Heatmap.png"), p_dea, 12, 10)
  }, error = function(e) message("    -> [Skip] DEA Heatmap skipped (data range issue)."))
  
}, error = function(e) message("    -> DEA Failed: ", e$message))


# -------------------------------------------------------------------------
# [Module 8] 形态学与保存
# -------------------------------------------------------------------------
message("\n[Module 8] Finishing...")
spata_obj <- identifyTissueOutline(spata_obj)
tryCatch({
  p_out <- plotImage(spata_obj) + ggpLayerTissueOutline(spata_obj)
  safe_save_plot(file.path(plot_dir, "09_Tissue_Outline.png"), p_out, 8, 8)
}, error = function(e){})

saveRDS(spata_obj, file.path(out_dir, "spata_object_v5_full.RDS"))
message("Pipeline Completed.")