# ===========================================================================
# File Name: step3_visualization_fix.R
# Author: Kuroneko
# Date: 2026-02-02
# Version: V2.0 (Robust Spatial Visualization via DimReduc)
#
# Description:
#     [Seurat Advanced Visualization - Robust Fix]
#     This script resolves the 'UpdateSeuratObject' error by treating spatial 
#     coordinates as a custom Dimensional Reduction ('spatial').
#     
#     This approach bypasses the strict validation of the 'SpatialImage' class
#     in Seurat v5, allowing for stable plotting of spatial distributions.
#
#     Key Functions:
#     1. Load Scanpy output (matrix, metadata, coordinates).
#     2. Inject spatial coordinates as a reduction (Fixes the crash).
#     3. Identify marker genes (FindAllMarkers).
#     4. Generate Top5 DotPlot (Bubble Plot).
#     5. Generate Top5 Heatmap.
#     6. Generate Spatial Cluster Map (using DimPlot).
#
# Input Files:
#     - mat.csv
#     - metadata.tsv
#     - position_spatial.tsv
#
# Output Files:
#     - 01_cluster_spatial_distribution.png
#     - 02_top5_markers_dotplot.png
#     - 03_top5_markers_heatmap.png
#     - top5_markers.csv
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
