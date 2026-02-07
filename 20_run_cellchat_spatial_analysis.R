# ===========================================================================
# File Name: 20_run_cellchat_spatial_analysis.R
# Author: Kuroneko
# Version: V1.0 (Spatial Enhanced Version)
#
# Description:
#     [CellChat 空间通讯分析 - 深度版]
#
#     核心功能:
#     1. 坐标映射: 结合 10x Visium 空间缩放因子计算物理距离。
#     2. 通讯概率计算: 基于 truncatedMean 过滤低丰度表达。
#     3. 空间距离约束: 通过 distance.use = TRUE 限制非相邻 spot 的通讯概率。
#     4. 多维可视化: 气泡图、弦图、以及在 H&E 图像上的信号流向映射。
#
# Input Files:
#     - Data Dir: F:/ST/code/data/V1_Human_Lymph_Node/
#     - H5 File: filtered_feature_bc_matrix.h5
#
# Output Files (in F:/ST/code/results/cellchat/):
#     - cellchat_spatial_obj.rds: 结果对象
#     - plots/: 包含所有通讯网络可视化图
# ===========================================================================
# BiocManager::install("glmGamPoi")
# -------------------------------------------------------------------------
# [Module 1] 环境准备与显存清理
# -------------------------------------------------------------------------
library(CellChat)
library(Seurat)
library(tidyverse)
library(patchwork)
library(jsonlite)
options(future.globals.maxSize = 8000 * 1024^2)
# 路径设置
BASE_DIR <- "F:/ST/code/"
DATA_DIR <- file.path(BASE_DIR, "data/V1_Human_Lymph_Node")
OUT_DIR <- file.path(BASE_DIR, "results/cellchat")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# 清理内存
gc()

# -------------------------------------------------------------------------
# [Module 2] 数据加载与预处理 (空间特异性)
# -------------------------------------------------------------------------
message("[Module 2] Loading Visium Data...")

# 读取 Seurat 对象
visium_obj <- Load10X_Spatial(DATA_DIR)
visium_obj <- SCTransform(visium_obj, assay = "Spatial", verbose = FALSE)

# 获取细胞类型标签
# 注意: 这里假设你已经完成了细胞类型注释，若未完成，可先用 Seurat 聚类代替
# 建议：使用反褶积后的结果填充 Idents(visium_obj)
# 这里先以 Seurat 聚类作为演示
visium_obj <- RunPCA(visium_obj) %>% FindNeighbors(dims = 1:30) %>% FindClusters(resolution = 0.5)

# -------------------------------------------------------------------------
# [Module 3] 提取空间参数 
# -------------------------------------------------------------------------
message("[Module 3] Extracting Spatial Information...")

# 1. 获取坐标 (Seurat V5 返回的是 data.frame，可能包含 extra columns)
# 使用 scale = NULL 获取原始图像像素坐标 (imagecol, imagerow)
spatial_locs <- GetTissueCoordinates(visium_obj, scale = NULL)

# 检查一下列名，Seurat V5 通常返回 c("x", "y") 或者 c("imagecol", "imagerow")
# 我们只取前两列，并强制转为矩阵
spatial_locs <- as.matrix(spatial_locs[, 1:2])

# 确保列名是 CellChat 喜欢的格式 (虽然它只检测列数，但这有助于后续绘图)
colnames(spatial_locs) <- c("imagerow", "imagecol")

# 检查一下修正后的数据
message("    Coordinate format check: ", paste(dim(spatial_locs), collapse = " x "))
message("    First few rows:")
print(head(spatial_locs))

# 2. 读取缩放因子计算实际距离
# 10X Visium spot 直径约为 55um，中心距离约为 100um
# CellChat 需要计算 spot 间的物理距离权重
scale_factors <- jsonlite::fromJSON(file.path(DATA_DIR, "spatial/scalefactors_json.json"))
spot_size <- 65 # 包含 gap 的有效直径
conversion_factor <- spot_size / scale_factors$spot_diameter_fullres
spatial_factors <- data.frame(ratio = conversion_factor, tol = spot_size / 2)


# -------------------------------------------------------------------------
# [Module 4] 构建 CellChat 对象 (Seurat V5 + 标签修正版)
# -------------------------------------------------------------------------
message("[Module 4] Building CellChat Object...")

# 1. 获取表达矩阵 (Seurat V5 写法)
data_input <- GetAssayData(visium_obj, layer = "data", assay = "SCT")

# 2. 构建元数据并修复 "0" 标签错误
# CellChat 不允许 Label 为 0，所以我们给所有聚类加个前缀 "C" (例如 C0, C1...)
original_idents <- Idents(visium_obj)
# 方法 A: 加前缀 (推荐)
new_labels <- paste0("C", original_idents) 
# 方法 B: 如果你已经有具体的细胞类型名字，确保它们不是纯数字 0

meta <- data.frame(labels = new_labels, row.names = names(original_idents))

# 3. 再次检查数据对齐
if(!all(colnames(data_input) == rownames(meta))) {
  stop("Error: Cell names in expression matrix and metadata do not match!")
}

# 4. 创建 CellChat 对象
cellchat <- createCellChat(
  object = data_input, 
  meta = meta, 
  group.by = "labels",
  datatype = "spatial",
  coordinates = spatial_locs,
  spatial.factors = spatial_factors
)

# 5. 设置数据库
CellChatDB <- CellChatDB.human 
cellchat@DB <- CellChatDB

message("    CellChat object created successfully. Labels: ", 
        paste(unique(meta$labels), collapse=", "))
# -------------------------------------------------------------------------
# [Module 5] 运行通讯推断 (修复参数名)
# -------------------------------------------------------------------------
message("[Module 5] Inferring Cell-Cell Communication...")

# 预处理
cellchat <- subsetData(cellchat)
future::plan("multisession", workers = 4) # 开启多线程

cellchat <- identifyOverExpressedGenes(cellchat)
cellchat <- identifyOverExpressedInteractions(cellchat)

# [FIX] 修复参数名错误: interaction.range -> contact.range
# 计算通讯概率 (启用空间约束)
future::plan("sequential")

cellchat <- computeCommunProb(
  cellchat, 
  type = "truncatedMean", trim = 0.1,
  distance.use = TRUE, 
  contact.range = 250, # 注意：这里必须用 contact.range
  scale.distance = 0.01
)

cellchat <- filterCommunication(cellchat, min.cells = 10)
cellchat <- computeCommunProbPathway(cellchat)
cellchat <- aggregateNet(cellchat)

message("    Communication inference completed.")
# -------------------------------------------------------------------------
# [Module 6] 多维可视化
# -------------------------------------------------------------------------
message("[Module 6] Generating Visualizations...")

# 1. 整体网络交互强度图
pdf(file.path(OUT_DIR, "01_Interaction_Overview.pdf"), width = 10, height = 5)
par(mfrow = c(1,2), xpd=TRUE)
netVisual_circle(cellchat@net$count, weight.scale = T, title.name = "Number of interactions")
netVisual_circle(cellchat@net$weight, weight.scale = T, title.name = "Interaction strength")
dev.off()

# 2. 空间流向图 (以排名第一的通路为例)
pathways_all <- cellchat@netP$pathways
target_pathway <- pathways_all[1]

pdf(file.path(OUT_DIR, paste0("02_Spatial_Net_", target_pathway, ".pdf")), width = 8, height = 8)
netVisual_aggregate(
  cellchat,
  signaling = target_pathway,
  layout = "spatial",
  edge.width.max = 2,
  vertex.size.max = 3,
  alpha.image = 0.3
)
dev.off()

# -------------------------------------------------------------------------
# [Module 7] 保存结果
# -------------------------------------------------------------------------
saveRDS(cellchat, file = file.path(OUT_DIR, "cellchat_spatial_results.rds"))

message("CellChat Analysis Completed. Files saved in: ", OUT_DIR)
