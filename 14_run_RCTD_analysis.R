# ===========================================================================
# File Name: run_RCTD_analysis_v2.1.R
# Author: Kuroneko
# Date: 2026-02-02
# Version: V2.1 (Fix Package Loading & Seurat v5)
#
# Description:
#     [RCTD 细胞类型反卷积全流程 - 稳健修复版]
#     
#     Update V2.1:
#     - Added STRICT package loading check. If 'spacexr' is not found, 
#       the script will STOP immediately to warn the user.
#     - Maintained Seurat v5 compatibility (layer="counts").
#     
#     核心步骤:
#     1. [Reference]: 加载单细胞数据，进行质控、标准化，构建 RCTD Reference 对象。
#     2. [Spatial]: 加载 10x Visium 空间数据，构建 SpatialRNA 对象。
#     3. [Deconvolution]: 运行 RCTD (Doublet Mode) 推断每个 Spot 的细胞类型组成。
#     4. [Integration]: 将 RCTD 结果（权重、主导细胞类型）整合回 Seurat 对象。
#     5. [Visualization]: 绘制空间权重图、饼图 (Scatterpies) 和 Seurat 风格展示图。
#
# Input Files:
#     - Single Cell: F:/ST/code/05/sc/brain_sc_expression_matrix.txt
#     - Single Cell Meta: F:/ST/code/05/sc/brain_sc_metadata.csv
#     - Spatial: F:/ST/code/05/10x_brain/
#
# Output Files (in F:/ST/code/05/RCTD_results_custom/):
#     - RCTD_results.rds (保存完整的 RCTD 对象)
#     - spatial_seurat_with_rctd.rds (整合了结果的 Seurat 对象)
#     - 01_spatial_weights_*.png (各细胞类型权重图)
#     - 02_spatial_scatterpies.png (空间饼图)
#     - 03_seurat_spatial_feature.png (Seurat 风格展示)
# ===========================================================================
# -------------------------------------------------------------------------
# [Module 1] 环境初始化 (严格检查)
# -------------------------------------------------------------------------
message("[Module 1] Initializing Environment...")

Sys.setenv(LANGUAGE = "en")
options(stringsAsFactors = FALSE)
rm(list=ls())

packages <- c("spacexr", "Matrix", "doParallel", "ggplot2", "Seurat", "data.table", 
              "tidyverse", "patchwork", "STdeconvolve")

for(pkg in packages){
  if(!require(pkg, character.only = T)){
    stop(paste0("[Critical Error] Package '", pkg, "' is NOT installed."))
  }
}

BASE_DIR <- "F:/ST/code/05/"
OUT_DIR <- file.path(BASE_DIR, "RCTD_results_custom")
if(!dir.exists(OUT_DIR)) dir.create(OUT_DIR)
setwd(OUT_DIR)

# -------------------------------------------------------------------------
# [Module 2] 准备单细胞参考数据 (Reference)
# -------------------------------------------------------------------------
message("\n[Module 2] Preparing Single-Cell Reference...")

# 1. 读取单细胞数据
sc_exp_path <- file.path(BASE_DIR, "sc/brain_sc_expression_matrix.txt")
message("    Reading SC Expression Matrix...")
if(!file.exists(sc_exp_path)) stop("Single cell matrix file not found!")
exp <- fread(sc_exp_path, data.table = F)
rownames(exp) <- exp[,1]
exp <- exp[,-1]

# 2. 读取元数据
sc_meta_path <- file.path(BASE_DIR, "sc/brain_sc_metadata.csv")
message("    Reading SC Metadata...")
if(!file.exists(sc_meta_path)) stop("Single cell metadata file not found!")
meta <- read.csv(sc_meta_path)
rownames(meta) <- meta$X 

# 3. 构建 Seurat 对象
message("    Creating Seurat Object for QC...")
sc_obj <- CreateSeuratObject(counts = exp, meta.data = meta)
sc_obj[["percent.mt"]] <- PercentageFeatureSet(sc_obj, pattern = "^mt-")
sc_obj <- subset(sc_obj, subset = nCount_RNA > 1000 & nFeature_RNA > 200 & percent.mt < 25)
sc_obj <- NormalizeData(sc_obj, normalization.method = "LogNormalize", scale.factor = 10000)

# 4. 构建 RCTD Reference
message("    Extracting counts for Reference...")
# [Seurat v5] 使用 layer="counts"
counts <- GetAssayData(sc_obj, layer = "counts")

if("Class" %in% colnames(sc_obj@meta.data)){
  cell_types <- sc_obj$Class
} else {
  stop("Error: 'Class' column not found in SC metadata.")
}
names(cell_types) <- colnames(sc_obj)
cell_types <- as.factor(cell_types)
nUMI <- colSums(counts)

message(paste("    Building Reference with", length(levels(cell_types)), "cell types..."))
reference <- Reference(counts, cell_types, nUMI)

# -------------------------------------------------------------------------
# [Module 3] 准备空间数据 (Spatial) - [CRITICAL FIX HERE]
# -------------------------------------------------------------------------
message("\n[Module 3] Preparing Spatial Data...")

spatial_dir <- file.path(BASE_DIR, "10x_brain")

# 1. 使用 Seurat 读取
message("    Loading Spatial Data via Seurat...")
if(!dir.exists(spatial_dir)) stop("Spatial data directory not found!")
spatial_obj <- Load10X_Spatial(spatial_dir)

# 2. 转换为 RCTD SpatialRNA 对象
message("    Converting to SpatialRNA object...")
counts_sp <- GetAssayData(spatial_obj, assay = "Spatial", layer = "counts")

# [CRITICAL FIX] 严格清洗坐标矩阵
# Seurat v5 GetTissueCoordinates 可能返回多余列 (如 tissue, row, col)
raw_coords <- GetTissueCoordinates(spatial_obj)

# 优先查找 x 和 y 列
if(all(c("x", "y") %in% colnames(raw_coords))) {
  message("    Found 'x' and 'y' columns. Filtering...")
  coords <- raw_coords[, c("x", "y")]
} else if(all(c("imagecol", "imagerow") %in% colnames(raw_coords))) {
  message("    Found 'imagecol' and 'imagerow'. Renaming...")
  coords <- raw_coords[, c("imagecol", "imagerow")]
  colnames(coords) <- c("x", "y")
} else {
  # 最后的手段：只取前两列，并强制重命名
  message("    Column names match neither standard. Using first 2 columns...")
  coords <- raw_coords[, 1:2]
  colnames(coords) <- c("x", "y")
}

# 确保坐标是数值型
coords$x <- as.numeric(coords$x)
coords$y <- as.numeric(coords$y)

# 再次检查维度
if(ncol(coords) != 2) stop(paste("Error: Coords still has", ncol(coords), "columns. Must be 2."))

nUMI_sp <- colSums(counts_sp)

puck <- SpatialRNA(coords, counts_sp, nUMI_sp)

common_genes <- intersect(rownames(reference@counts), rownames(puck@counts))
message(paste("    Found", length(common_genes), "common genes."))

# -------------------------------------------------------------------------
# [Module 4] 运行 RCTD (Deconvolution)
# -------------------------------------------------------------------------
message("\n[Module 4] Running RCTD...")

# 1. 创建 RCTD 对象
myRCTD <- create.RCTD(puck, reference, max_cores = 4)

# 2. 运行 RCTD (Doublet Mode)
message("    Executing run.RCTD (mode = 'doublet')... This may take a while.")
myRCTD <- run.RCTD(myRCTD, doublet_mode = 'doublet')

# 3. 保存结果
saveRDS(myRCTD, file.path(OUT_DIR, "RCTD_results.rds"))
message("    RCTD run complete. Results saved.")

# -------------------------------------------------------------------------
# [Module 5] 结果整合与解析
# -------------------------------------------------------------------------
message("\n[Module 5] Integrating Results into Seurat...")

results <- myRCTD@results
norm_weights <- normalize_weights(results$weights)

# 添加权重信息
spatial_obj <- AddMetaData(spatial_obj, metadata = as.data.frame(norm_weights))
# 添加主导细胞类型信息
spatial_obj <- AddMetaData(spatial_obj, metadata = results$results_df)

saveRDS(spatial_obj, file.path(OUT_DIR, "spatial_seurat_with_rctd.rds"))

# -------------------------------------------------------------------------
# [Module 6] 高级可视化
# -------------------------------------------------------------------------
message("\n[Module 6] Generating Plots...")

# 1. 空间权重分布图
message("    Plotting Spatial Weights...")
cell_types_to_plot <- colnames(norm_weights)
selected_types <- cell_types_to_plot[1:min(4, length(cell_types_to_plot))] 

p1 <- SpatialFeaturePlot(spatial_obj, features = selected_types, 
                         pt.size.factor = 1.6, ncol = 2, crop = FALSE) +
  plot_annotation(title = "RCTD Estimated Proportions")
ggsave(file.path(OUT_DIR, "01_spatial_weights_top4.png"), plot = p1, width = 12, height = 10)

# 2. 主导细胞类型分布图
message("    Plotting Dominant Cell Types...")
p2 <- SpatialDimPlot(spatial_obj, group.by = "first_type", pt.size.factor = 1.6) +
  ggtitle("Dominant Cell Type (RCTD Doublet Mode)")
ggsave(file.path(OUT_DIR, "02_dominant_cell_type.png"), plot = p2, width = 10, height = 8)

# 3. 散点饼图 (Scatterpies)
message("    Plotting Scatterpies...")
tryCatch({
  pos_df <- GetTissueCoordinates(spatial_obj)
  # 同样对绘图用的坐标进行清洗
  if(all(c("x", "y") %in% colnames(pos_df))) {
    pos_df <- pos_df[, c("x", "y")]
  } else {
    pos_df <- pos_df[, 1:2]
    colnames(pos_df) <- c("x", "y")
  }
  
  plt <- vizAllTopics(theta = as.matrix(norm_weights),
                      pos = pos_df,
                      topicOrder = seq(ncol(norm_weights)),
                      topicCols = rainbow(ncol(norm_weights)),
                      groups = NA,
                      group_cols = NA,
                      r = 1.5, 
                      lwd = 0.1,
                      showLegend = TRUE,
                      plotTitle = "Cell Type Proportions (Scatterpies)")
  
  ggsave(file.path(OUT_DIR, "03_spatial_scatterpies.png"), plot = plt, width = 12, height = 10)
}, error = function(e){
  message("    [Warning] Scatterpie plotting failed: ", e$message)
})

# 4. 特定细胞类型共定位展示
if(all(c("Oligos", "Astrocytes") %in% colnames(spatial_obj@meta.data))) {
  message("    Plotting Co-localization (Oligos vs Astrocytes)...")
  p4 <- SpatialFeaturePlot(spatial_obj, features = c("Oligos", "Astrocytes"), 
                           pt.size.factor = 1.6, alpha = c(0.1, 1), ncol = 2)
  ggsave(file.path(OUT_DIR, "04_colocalization_example.png"), plot = p4, width = 12, height = 6)
}

message("RCTD Pipeline Completed Successfully.")
message("Results saved to: ", OUT_DIR)




# 1. 准备数据
norm_weights <- spatial_obj@meta.data[, colnames(spatial_obj@meta.data) %in% levels(spatial_obj$first_type)]
# 只要权重列（过滤掉其他 metadata）
# 或者更简单的，直接从之前保存的 myRCTD 对象里取，或者只取数值列
norm_weights <- spatial_obj@meta.data[, grep("^[A-Z]", colnames(spatial_obj@meta.data))] # 假设细胞类型是大写开头
# 为了保险，我们只取前几列纯数值的权重列（你需要根据实际列名调整）
# 这里我们用一个更通用的方法：找到所有数值列且行和为1左右的列
numeric_cols <- sapply(spatial_obj@meta.data, is.numeric)
potential_weights <- spatial_obj@meta.data[, numeric_cols]
# 通常 RCTD 的权重列名就是细胞类型名

pos_df <- GetTissueCoordinates(spatial_obj)
if(all(c("x", "y") %in% colnames(pos_df))) {
  pos_df <- pos_df[, c("x", "y")]
} else {
  pos_df <- pos_df[, 1:2]
  colnames(pos_df) <- c("x", "y")
}

# 2. 尝试不同的半径 (r) 重新画图
# 10x Visium 的点直径通常很大，尝试 r = 100
r_size <- 100 

plt <- vizAllTopics(theta = as.matrix(norm_weights),
                    pos = pos_df,
                    topicOrder = seq(ncol(norm_weights)),
                    topicCols = rainbow(ncol(norm_weights)),
                    groups = NA,
                    group_cols = NA,
                    r = r_size,   # <--- 关键修改在这里！把 1.5 改成 100 或者 200
                    lwd = 0.1,
                    showLegend = TRUE,
                    plotTitle = "Cell Type Proportions (Scatterpies)")

ggsave("F:/ST/code/05/RCTD_results_custom/03_spatial_scatterpies_FIXED.png", plot = plt, width = 12, height = 10)
print("新图已保存，快去看看有没有点！")
##建议：先试 r = 100，如果圆圈重叠太厉害看不清，就改小点
