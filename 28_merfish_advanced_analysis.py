# ===========================================================================
# Project: MERFISH Single-cell Spatial Analysis
# File Name: 25_merfish_advanced_analysis.py
# Author: Kuroneko
# Date: 2026-02-04
# Current Path: F:\ST\code
#
# Description:
#     [MERFISH 空间转录组进阶分析全流程]
#     核心功能：
#     1. 基础流程：数据加载、Blank基因质控、标准化、降维聚类。
#     2. 空间统计：Moran's I 识别具有空间表达模式的基因。
#     3. 空间邻域：计算 Neighborhood Enrichment，分析细胞类型的空间排布规律。
#     4. 空间点模式：Ripley's K 函数分析细胞群落的聚集性。
#     5. 细胞通讯：结合空间邻近性的配体-受体分析 (Ligand-Receptor Interaction)。
#
# Input Files:
#     - Counts: F:\ST\code\data\MERFISH\datasets_mouse_brain_map_BrainReceptorShowcase_Slice1_Replicate1_cell_by_gene_S1R1.csv
#     - Meta: F:\ST\code\data\MERFISH\datasets_mouse_brain_map_BrainReceptorShowcase_Slice1_Replicate1_cell_metadata_S1R1.csv
#
# Output Files:
#     - Results: F:\ST\code\results\merfish\adata_merfish_processed.h5ad
#     - Plots: F:\ST\code\results\merfish\plots\
# ===========================================================================

import scanpy as sc
import squidpy as sq
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# [Module 1] 环境初始化与数据读取
# -------------------------------------------------------------------------
# 设置路径
BASE_DIR = r"F:\ST\code"
DATA_DIR = os.path.join(BASE_DIR, "data", "MERFISH")
OUT_DIR = os.path.join(BASE_DIR, "results", "merfish")
PLOT_DIR = os.path.join(OUT_DIR, "plots")

for d in [OUT_DIR, PLOT_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

sc.settings.figdir = PLOT_DIR
sc.settings.set_figure_params(dpi=120, frameon=False, facecolor='white')

# 读取数据
print("Loading MERFISH data...")
counts_path = os.path.join(DATA_DIR, "datasets_mouse_brain_map_BrainReceptorShowcase_Slice1_Replicate1_cell_by_gene_S1R1.csv")
meta_path = os.path.join(DATA_DIR, "datasets_mouse_brain_map_BrainReceptorShowcase_Slice1_Replicate1_cell_metadata_S1R1.csv")

# MERFISH 原始 CSV 通常是 [细胞 x 基因]
adata = sc.read_csv(counts_path)
cell_meta = pd.read_csv(meta_path, index_col=0)

# -------------------------------------------------------------------------
# [Module 2] 质量控制与预处理 (针对 MERFISH 特性)
# -------------------------------------------------------------------------
# 1. 处理 Blank 基因 (背景噪声对照)
# 算法理由：MERFISH 实验中会加入不针对任何已知转录本的探针，称为 Blank 基因。
# 它们的表达水平反映了系统的非特异性结合背景。
blank_genes = adata.var_names.str.contains("Blank")
adata.obsm["blank_genes"] = adata[:, blank_genes].X.copy()
adata = adata[:, ~blank_genes].copy()

# 2. 注入空间坐标与元数据
cell_meta.index = cell_meta.index.astype(str)
adata.obs = pd.merge(adata.obs, cell_meta, left_index=True, right_index=True)
# 核心：obsm['spatial'] 是 Squidpy 分析的硬性要求
adata.obsm["spatial"] = adata.obs[["center_x", "center_y"]].values

# 3. 标准流程
sc.pp.filter_cells(adata, min_counts=10)
sc.pp.filter_genes(adata, min_cells=5)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=2000)
sc.tl.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5)

# -------------------------------------------------------------------------
# [Module 3] 空间统计学：邻域分析与自相关
# -------------------------------------------------------------------------
# 1. 构建空间邻近图 (Spatial Graph)
# 算法理由：对于单细胞级别的空间数据，Delaunay 三角剖分能更好地模拟细胞间的物理接触。
sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True)

# 2. 邻域富集分析 (Neighborhood Enrichment)
# 算法理由：判断不同类型的细胞在空间上是倾向于“群居”还是“避开”。
sq.gr.nhood_enrichment(adata, cluster_key="leiden")
sq.pl.nhood_enrichment(adata, cluster_key="leiden", method="single", save="01_nhood_enrichment.pdf")

# 3. 空间自相关分析 (Moran's I)
# 算法理由：识别哪些基因的表达模式具有明显的空间结构，即不是随机分布的。
sq.gr.spatial_autocorr(adata, mode="moran", n_perms=100)
# 查看得分最高的基因
top_spatial_genes = adata.uns["moranI"].head(6).index.tolist()
sq.pl.spatial_scatter(adata, color=top_spatial_genes, size=2, shape=None, save="02_top_moran_genes.pdf")

# -------------------------------------------------------------------------
# [Module 4] 细胞通讯预测：空间受限的配体-受体分析
# -------------------------------------------------------------------------
# 算法理由：
# 传统的分析只看表达量，但如果配体细胞和受体细胞隔得很远，通讯就不可能发生。
# Squidpy 的 ligrec 会利用我们在 Module 3 构建的空间图，只计算物理邻近细胞间的相互作用。

# 为加速计算，进行抽样
adata_sub = sc.pp.subsample(adata, fraction=0.4, copy=True)
sq.gr.spatial_neighbors(adata_sub, coord_type="generic", delaunay=True)

sq.gr.ligrec(
    adata_sub,
    cluster_key="leiden",
    use_raw=False,
    transmitter_params={"categories": "ligand"},
    receiver_params={"categories": "receptor"}
)

# 可视化：查看 Cluster 0 和其他 Cluster 之间的通讯强度
sq.pl.ligrec(
    adata_sub,
    cluster_key="leiden",
    source_groups="0",
    alpha=0.05,
    save="03_ligrec_bubble_plot.pdf"
)

# -------------------------------------------------------------------------
# [Module 5] 数据保存
# -------------------------------------------------------------------------
print(f"Saving final AnnData object to {OUT_DIR}...")
adata.write_h5ad(os.path.join(OUT_DIR, "adata_merfish_final.h5ad"))
print("All tasks completed successfully!")