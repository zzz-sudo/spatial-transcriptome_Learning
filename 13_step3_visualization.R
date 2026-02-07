# ===========================================================================
# 文件名称: step3_visualization_fix.R
# 作    者: Kuroneko
# 日    期: 2026-02-02
# 版    本: V2.0 (基于降维对象的稳健空间可视化)
#
# 功能描述:
#     [Seurat 高级可视化 - 稳健修复版]
#     本脚本通过将空间物理坐标 (x, y) 封装为自定义的“降维结果”('spatial')，
#     彻底解决了 Seurat v5 中 'UpdateSeuratObject' 及 SpatialImage 对象构建的报错问题。
#     
#     该方法绕过了 Seurat 对 SpatialImage 类的严格校验，
#     实现了稳定、兼容的空间分布绘图。
#
#     核心功能:
#     1. 加载 Scanpy 导出的数据 (表达矩阵、元数据、坐标)。
#     2. 坐标注入: 将物理坐标作为降维嵌入 (核心修复)。
#     3. 差异分析: 计算 Marker 基因 (FindAllMarkers)。
#     4. 生成 Top5 气泡图 (DotPlot)。
#     5. 生成 Top5 热图 (Heatmap)。
#     6. 生成空间聚类分布图 (使用 DimPlot)。
#
# 输入文件:
#     - mat.csv (表达矩阵)
#     - metadata.tsv (元数据)
#     - position_spatial.tsv (空间坐标)
#
# 输出文件:
#     - 01_cluster_spatial_distribution.png (空间聚类图)
#     - 02_top5_markers_dotplot.png (Marker 气泡图)
#     - 03_top5_markers_heatmap.png (Marker 热图)
#     - top5_markers.csv (差异基因列表)
# ===========================================================================
# 1. Environment Setup
cat("[Module 1] Initializing R Environment...\n")
Sys.setenv(LANGUAGE = "en")
options(stringsAsFactors = FALSE)

# Check libraries
required_packages <- c("Seurat", "dplyr", "Matrix", "ggplot2", "patchwork")
for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE)) {
    stop(paste("[Error] Package not found:", pkg))
  }
}

# Set Working Directory
work_dir <- "F:/ST/code/results/scanpy_seurat/"
if (!dir.exists(work_dir)) {
  stop(paste("[Error] Directory not found:", work_dir))
}
setwd(work_dir)

# 2. Load Data
cat("\n[Module 2] Loading Data...\n")

# Load Matrix
if (!file.exists("mat.csv")) stop("mat.csv not found")
cat("    Reading Expression Matrix...\n")
mydata <- read.csv("mat.csv")
rownames(mydata) <- mydata[,1]
mydata <- mydata[,-1]
mat <- Matrix(t(mydata), sparse = TRUE)

# Load Metadata
cat("    Reading Metadata...\n")
meta <- read.table("metadata.tsv", sep="\t", header=TRUE, row.names=1)

# Load Coordinates
cat("    Reading Coordinates...\n")
pos <- read.table("position_spatial.tsv", sep="\t", header=TRUE, row.names=1)

# 3. Construct Seurat Object with Spatial Reduction (The FIX)
cat("\n[Module 3] Constructing Object & Injecting Coordinates...\n")

# Create base object
obj <- CreateSeuratObject(counts = mat, project = 'Spatial', assay = 'Spatial', meta.data = meta)

# CRITICAL FIX: Add coordinates as a custom Dimensional Reduction
# This tricks Seurat into treating X/Y coordinates like UMAP coordinates
# Filter coordinates to match cells in object
common_cells <- intersect(colnames(obj), rownames(pos))
obj <- subset(obj, cells = common_cells)
pos <- pos[common_cells, ]

# Create embedding matrix
spatial_embed <- as.matrix(pos[, c("x", "y")])
colnames(spatial_embed) <- c("Spatial_1", "Spatial_2")

# Add 'spatial' reduction
obj[["spatial"]] <- CreateDimReducObject(embeddings = spatial_embed, key = "Spatial_", assay = "Spatial")
cat("    Spatial coordinates successfully added as 'spatial' reduction.\n")

# 4. Processing & Analysis
cat("\n[Module 4] Normalization & Differential Expression...\n")

# Normalize and Scale
obj <- NormalizeData(obj, verbose = FALSE)
obj <- ScaleData(obj, verbose = FALSE)

# Set Identity (Use 'clusters' from Python)
if ("clusters" %in% colnames(obj@meta.data)) {
  Idents(obj) <- "clusters"
  cat("    Identity set to 'clusters'.\n")
} else {
  stop("[Error] 'clusters' column missing in metadata.")
}

# Find Markers
cat("    Finding Top 5 Markers per Cluster (this may take a minute)...\n")
markers <- FindAllMarkers(obj, only.pos = TRUE, min.pct = 0.25, logfc.threshold = 0.25, verbose = FALSE)

# Extract Top 5
if (nrow(markers) > 0) {
  top5 <- markers %>% group_by(cluster) %>% top_n(n = 5, wt = avg_log2FC)
  write.csv(top5, "top5_markers.csv")
  cat("    Markers saved to CSV.\n")
} else {
  stop("[Error] No markers found. Check normalization or data quality.")
}

# 5. Visualization (Robust Methods)
cat("\n[Module 5] Generating Plots...\n")

# (1) Spatial Distribution Plot
# We use DimPlot with reduction='spatial' instead of SpatialDimPlot
# scale_y_reverse is added because image coordinates usually have inverted Y axis compared to plots
cat("    Plotting 01_cluster_spatial_distribution.png...\n")
p1 <- DimPlot(obj, reduction = "spatial", label = TRUE, label.size = 5, pt.size = 1.5) + 
  ggtitle("Spatial Distribution of Clusters") +
  scale_y_reverse() + 
  theme_void() + # Removes axes to look like a spatial plot
  theme(plot.title = element_text(hjust = 0.5, size = 16),
        legend.position = "right")
ggsave("01_cluster_spatial_distribution.png", plot = p1, width = 8, height = 8)

# (2) DotPlot (Bubble Plot)
cat("    Plotting 02_top5_markers_dotplot.png...\n")
unique_genes <- unique(top5$gene)
p2 <- DotPlot(obj, features = unique_genes) + 
  RotatedAxis() + 
  ggtitle("Top 5 Markers per Cluster") +
  theme(axis.text.x = element_text(size = 8),
        plot.title = element_text(hjust = 0.5))
ggsave("02_top5_markers_dotplot.png", plot = p2, width = 14, height = 8)

# (3) Heatmap
cat("    Plotting 03_top5_markers_heatmap.png...\n")
p3 <- DoHeatmap(obj, features = top5$gene) + 
  ggtitle("Expression Heatmap") +
  theme(plot.title = element_text(hjust = 0.5))
ggsave("03_top5_markers_heatmap.png", plot = p3, width = 12, height = 10)
cat("Visualization Pipeline Completed Successfully.\n")
cat("Output saved to:", work_dir, "\n")

