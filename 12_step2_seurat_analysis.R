# ===========================================================================
# 文件名称: step2_seurat_analysis_final_v3.R
# 作    者: Kuroneko
# 版    本: V3.0 (稳健空间可视化版 - 绕过 VisiumV1 类)
#
# 功能描述:
#     [Seurat 空间转录组分析 - 稳健流程]
#     本脚本通过绕过 Seurat v5 中脆弱的 'VisiumV1' 对象构建，
#     解决了 "UpdateSeuratObject" 等常见的版本兼容性报错。
#
#     核心方法:
#     我们不构建复杂的 SpatialImage 对象，而是将空间物理坐标 (x, y) 
#     直接存储为一个自定义的降维结果 ('spatial')。
#     这使得我们可以使用标准的 DimPlot/FeaturePlot 函数直接进行空间可视化，
#     且完全不受 Seurat 版本升级的影响。
#
#     主要模块:
#     1. 数据加载 (读取 Python 导出的矩阵、元数据、坐标)。
#     2. 对象构建 (CreateSeuratObject)。
#     3. 空间注入 (将坐标转换为 'spatial' 降维对象)。
#     4. 差异分析 (FindAllMarkers 计算 Marker 基因)。
#     5. 高级可视化 (空间聚类图、热图、气泡图、基因分布图)。
#
# 输入文件 (来自 Python step1 的输出):
#     - mat.csv (表达矩阵)
#     - metadata.tsv (元数据，含 clusters)
#     - position_spatial.tsv (空间坐标)
#
# 输出文件:
#     - 01_spatial_clusters.png (空间聚类图)
#     - 02_spatial_gene_counts.png (UMI 计数分布)
#     - 03_heatmap.png (差异基因热图)
#     - 04_dotplot.png (差异基因气泡图)
#     - 05_gene_expression.png (Marker 基因空间分布)
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
cat("    Reading Expression Matrix (mat.csv)...\n")
if (!file.exists("mat.csv")) stop("mat.csv missing")
mydata <- read.csv("mat.csv")
rownames(mydata) <- mydata[,1]
mydata <- mydata[,-1]
mat <- Matrix(t(mydata), sparse = TRUE) # Gene x Cell

# Load Metadata
cat("    Reading Metadata (metadata.tsv)...\n")
meta <- read.table("metadata.tsv", sep="\t", header=TRUE, row.names=1)

# Load Coordinates
cat("    Reading Coordinates (position_spatial.tsv)...\n")
pos <- read.table("position_spatial.tsv", sep="\t", header=TRUE, row.names=1)

# 3. Create Seurat Object
cat("\n[Module 3] Constructing Seurat Object...\n")
obj <- CreateSeuratObject(counts = mat, project = 'Spatial', assay = 'Spatial', meta.data = meta)

# ---------------------------------------------------------------------------
# [CRITICAL FIX] Robust Spatial Embedding Strategy
# Instead of making a fragile VisiumV1 object, we create a DimReduc object.
# This forces Seurat to treat (x,y) coordinates as a layout like UMAP.
# ---------------------------------------------------------------------------
cat("\n[Module 4] Injecting Spatial Coordinates as Reduction...\n")

# Prepare coordinates matrix
# Ensure cell names match
common_cells <- intersect(colnames(obj), rownames(pos))
pos <- pos[common_cells, ]
obj <- subset(obj, cells = common_cells)

# Create a matrix for embedding
spatial_embed <- as.matrix(pos[, c("x", "y")])
colnames(spatial_embed) <- c("Spatial_1", "Spatial_2")

# Add to Seurat object as a custom reduction named 'spatial'
obj[["spatial"]] <- CreateDimReducObject(embeddings = spatial_embed, key = "Spatial_", assay = "Spatial")

cat("    Spatial coordinates stored in obj[['spatial']].\n")

# 5. Analysis
cat("\n[Module 5] Analysis & Normalization...\n")
obj <- NormalizeData(obj, verbose = FALSE)
obj <- ScaleData(obj, verbose = FALSE)

# Set identity to Python clusters
if ("clusters" %in% colnames(obj@meta.data)) {
  Idents(obj) <- "clusters"
} else {
  cat("    [Warning] 'clusters' column missing. Using default identity.\n")
}

# 6. Visualization (Robust Mode)
cat("\n[Module 6] Generating Plots...\n")

# (1) Spatial Cluster Plot
# We use DimPlot but specify reduction='spatial' to plot on x,y coordinates
cat("    Plotting 01_spatial_clusters.png...\n")
p1 <- DimPlot(obj, reduction = "spatial", label = TRUE, label.size = 4, pt.size = 1.5) + 
  ggtitle("Spatial Clusters") +
  theme_void() + # Remove axes for cleaner spatial look
  theme(plot.title = element_text(hjust = 0.5)) +
  scale_y_reverse() # Spatial coordinates often need y-inversion to match image
ggsave("01_spatial_clusters.png", plot = p1, width = 8, height = 8)

# (2) Spatial Feature Plot (Total Counts)
cat("    Plotting 02_spatial_gene_counts.png...\n")
p2 <- FeaturePlot(obj, features = "total_counts", reduction = "spatial", pt.size = 1.5) +
  scale_colour_viridis_c(option = "magma") +
  ggtitle("Total RNA Counts") +
  theme_void() +
  scale_y_reverse()
ggsave("02_spatial_gene_counts.png", plot = p2, width = 8, height = 8)

# (3) Find Markers
cat("    Finding Markers (FindAllMarkers)...\n")
markers <- FindAllMarkers(obj, only.pos = TRUE, min.pct = 0.25, logfc.threshold = 0.25, verbose = FALSE)
top5 <- markers %>% group_by(cluster) %>% top_n(n = 5, wt = avg_log2FC)

# (4) Heatmap
cat("    Plotting 03_heatmap.png...\n")
if (nrow(top5) > 0) {
  p3 <- DoHeatmap(obj, features = top5$gene) + NoLegend()
  ggsave("03_heatmap.png", plot = p3, width = 12, height = 10)
}

# (5) Dot Plot
cat("    Plotting 04_dotplot.png...\n")
if (nrow(top5) > 0) {
  unique_top_genes <- unique(top5$gene)
  p4 <- DotPlot(obj, features = unique_top_genes) + 
    RotatedAxis() + 
    ggtitle("Top Marker Genes")
  ggsave("04_dotplot.png", plot = p4, width = 14, height = 8)
}

# (6) Gene Expression Spatial Plot
cat("    Plotting 05_gene_expression.png...\n")
if (nrow(top5) > 0) {
  target_genes <- top5$gene[1:min(3, length(top5$gene))]
  p5 <- FeaturePlot(obj, features = target_genes, reduction = "spatial", ncol = 3, pt.size = 1.5) & 
    scale_colour_viridis_c() & 
    theme_void() & 
    scale_y_reverse()
  ggsave("05_gene_expression.png", plot = p5, width = 15, height = 5)
}

cat("Analysis Pipeline Completed.\n")

cat("All plots saved to:", work_dir, "\n")

