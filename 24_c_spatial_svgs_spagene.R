# ===========================================================================
# File Name: 24_c_spatial_svgs_spagene.R
# Author: Kuroneko
# Date: 2026-02-04
# Version: V3.0 (Final Stable Edition)
#
# Description:
#     [SpaGene 空间模式与配受体共定位分析]
#     1. SVG Detection: 识别空间高变基因。
#     2. Pattern Discovery: 识别 10 种空间表达模式。
#     3. LR Colocalization: 验证配体-受体空间共定位 (基于 CellChatDB)。
# ===========================================================================

library(Seurat)
library(tidyverse)
library(SpaGene)
library(pheatmap)
library(CellChat)
library(ggplot2)
library(gridExtra)
library(stringr)

# -------------------------------------------------------------------------
# [Module 1 & 2] 环境与数据加载
# -------------------------------------------------------------------------
BASE_DIR <- "F:/ST/code/"
DATA_DIR <- file.path(BASE_DIR, "data/V1_Human_Lymph_Node")
OUT_DIR <- file.path(BASE_DIR, "results/spagene")
PLOT_DIR <- file.path(OUT_DIR, "plots")

dir.create(PLOT_DIR, showWarnings = FALSE, recursive = TRUE)

message("[Module 2] Preparing Count Matrix...")
visium_obj <- Load10X_Spatial(DATA_DIR)
count_mat <- GetAssayData(visium_obj, layer = "counts", assay = "Spatial")
locs <- GetTissueCoordinates(visium_obj, scale = NULL)
location_mat <- as.data.frame(locs[, 1:2])
colnames(location_mat) <- c("x", "y")

# 基因过滤 (保留在 >10 个 spot 中表达的基因)
count_mat <- count_mat[Matrix::rowSums(count_mat > 0) > 10, ]

# -------------------------------------------------------------------------
# [Module 3 & 4] SVG 与 Pattern 识别
# -------------------------------------------------------------------------
message("[Module 3/4] Running SpaGene Patterns...")
spagene_res <- SpaGene(count_mat, location_mat)
pattern_res <- FindPattern(spagene_res, nPattern = 10)

# 保存模式图
pdf(file.path(PLOT_DIR, "01_Spatial_Patterns.pdf"), width = 12, height = 10)
PlotPattern(pattern_res, location_mat, pt.size = 1.5)
dev.off()

# 提取代表基因并绘热图
top5_genes <- apply(pattern_res$genepattern, 2, function(x) names(x)[order(x, decreasing = TRUE)][1:5])
write.csv(top5_genes, file.path(OUT_DIR, "pattern_top_genes.csv"))

# 热图绘制 (强制清理设备防止损坏)
while(!is.null(dev.list())) dev.off()
plot_mat <- pattern_res$genepattern[unique(as.vector(top5_genes)), ]
plot_mat <- plot_mat[apply(plot_mat, 1, var) > 0, ] # 剔除方差为0

pdf(file.path(PLOT_DIR, "02_Pattern_Heatmap.pdf"), width = 8, height = 10)
pheatmap(plot_mat, scale = "row", cluster_cols = FALSE, main = "Pattern Genes Heatmap")
dev.off()

# -------------------------------------------------------------------------
# [Module 5] 配受体共定位分析 (LR Analysis)
# -------------------------------------------------------------------------
message("[Module 5] Ligand-Receptor Colocalization...")

# 1. 准备配对数据
LR_db <- CellChatDB.human$interaction
LR_input <- LR_db[LR_db$ligand %in% rownames(count_mat) & LR_db$receptor %in% rownames(count_mat), c("ligand", "receptor")]

# 2. 运行分析
lr_res <- SpaGene_LR(count_mat, location_mat, LRpair = LR_input)

# 3. 解析结果 (从行名获取正确基因名)
split_names <- str_split_fixed(rownames(lr_res), "_", 2)
lr_res$ligand <- split_names[, 1]
lr_res$receptor <- split_names[, 2]
lr_ranked <- lr_res[order(lr_res$adj), ]
write.csv(lr_ranked, file.path(OUT_DIR, "spagene_lr_colocalization.csv"))

# 4. 绘图 (Top 3)
top_lrs <- head(lr_ranked, 3)
plot_list <- list()

for (i in 1:nrow(top_lrs)) {
  lig <- trimws(top_lrs$ligand[i])
  rec <- trimws(top_lrs$receptor[i])
  
  # 关键检查：如果基因总表达量为 0，plotLR 会崩溃，此处直接跳过
  if (sum(count_mat[lig,]) == 0 || sum(count_mat[rec,]) == 0) {
    message(paste0("    -> Skipping ", lig, "-", rec, " (Zero expression in plot mode)"))
    next
  }
  
  message(paste0("    Plotting: ", lig, " - ", rec))
  p <- plotLR(count_mat, location_mat, LRpair = c(lig, rec), alpha.min = 0, pt.size = 1)
  p <- p + ggtitle(paste0(lig, " - ", rec, "\nScore: ", signif(top_lrs$score[i], 3))) +
    theme(plot.title = element_text(hjust = 0.5, size = 10))
  
  plot_list[[length(plot_list) + 1]] <- p
}

# 统一保存 PDF
if (length(plot_list) > 0) {
  pdf(file.path(PLOT_DIR, "03_Top_LR_Colocalization.pdf"), width = 4 * length(plot_list), height = 4)
  do.call(grid.arrange, c(plot_list, ncol = length(plot_list)))
  dev.off()
}

message("DONE: SpaGene Analysis Completed.")