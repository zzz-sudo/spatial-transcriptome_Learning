# -*- coding: utf-8 -*-
"""
===========================================================================
文件名称: step1_scanpy_process.py
作    者: Kuroneko
版    本: V1.0 

功能描述:
    [Scanpy 空间转录组标准分析与导出流程]
    本脚本执行标准的空间转录组无监督分析，不依赖单细胞参考集。
    
    核心步骤:
    1. 质控与过滤 (计算线粒体比例，过滤低质量斑点/基因)。
    2. 归一化与对数变换 (Normalize & Log1p)。
    3. 降维与聚类 (PCA -> UMAP -> Leiden 聚类)。
    4. 可视化 (绘制质控分布图、UMAP 聚类图、空间聚类图)。
    5. 数据导出 (生成 csv/tsv 文件供 R 语言 Seurat 包后续分析)。

输入文件:
    - F:/ST/code/04/GBM1_spaceranger_out (SpaceRanger 输出目录)

输出文件 (results/scanpy_seurat/):
    - mat.csv (表达矩阵)
    - metadata.tsv (元数据，包含 clusters 聚类结果)
    - position_spatial.tsv (空间坐标)
    - qc_plot.png, umap_plot.png, spatial_plot.png (分析图表)
===========================================================================
"""

import os
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# 1. Setup Environment
print("[Module 1] Initializing...")
# Note: Path modified to match your file.txt structure
# Please ensure this path matches your actual mounting point if 'F:' is not accessible directly
DATA_DIR = r"F:\ST\code\04\GBM1_spaceranger_out"
WORK_DIR = r"F:\ST\code\results\scanpy_seurat"  # Output directory

if not os.path.exists(WORK_DIR):
    os.makedirs(WORK_DIR)

# Set scanpy settings
sc.settings.verbosity = 3
sc.settings.figdir = WORK_DIR
sc.set_figure_params(dpi=150, facecolor="white", vector_friendly=True)

# 2. Load Data
print("\n[Module 2] Loading Visium Data...")
if not os.path.exists(DATA_DIR):
    print(f"[Error] Data directory not found: {DATA_DIR}")
    exit(1)

adata = sc.read_visium(DATA_DIR)
adata.var_names_make_unique()
print(f"    Loaded data shape: {adata.shape}")

# 3. Quality Control
print("\n[Module 3] Quality Control...")
adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

# Plot QC
print("    Plotting QC metrics...")
fig, axs = plt.subplots(1, 4, figsize=(15, 4))
sns.histplot(adata.obs["total_counts"], kde=False, ax=axs[0])
axs[0].set_title("Total Counts")
sns.histplot(adata.obs["total_counts"][adata.obs["total_counts"] < 10000], kde=False, bins=40, ax=axs[1])
axs[1].set_title("Total Counts < 10k")
sns.histplot(adata.obs["n_genes_by_counts"], kde=False, bins=60, ax=axs[2])
axs[2].set_title("N Genes")
sns.histplot(adata.obs["n_genes_by_counts"][adata.obs["n_genes_by_counts"] < 4000], kde=False, bins=60, ax=axs[3])
axs[3].set_title("N Genes < 4k")
plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, "qc_plot.png"))
plt.close()

# Filter
print("    Filtering data...")
sc.pp.filter_cells(adata, min_counts=5000)
sc.pp.filter_cells(adata, max_counts=35000)
adata = adata[adata.obs["pct_counts_mt"] < 20].copy()
sc.pp.filter_genes(adata, min_cells=10)
print(f"    Cells after filtering: {adata.n_obs}")

# 4. Processing
print("\n[Module 4] Processing (Normalize, Log1p, PCA, UMAP)...")
sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)

sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.tl.leiden(adata, key_added="clusters", directed=False, n_iterations=2)

# 5. Visualization
print("\n[Module 5] Generating Plots...")
# UMAP
sc.pl.umap(adata, color=["total_counts", "n_genes_by_counts", "clusters"], wspace=0.4, show=False, save="_umap.png")

# Spatial
sc.pl.spatial(adata, img_key="hires", color=["total_counts", "clusters"], size=1.5, show=False, save="_spatial.png")

# 6. Export for R
print("\n[Module 6] Exporting Data for Seurat...")


def adata_info(adata, out_dir):
    # Export Matrix
    # Use sparse format is better but for compatibility with k10.md we use dense CSV
    # Be careful with large datasets
    print("    Exporting Matrix...")
    mat = pd.DataFrame(data=adata.X.todense(), index=adata.obs_names, columns=adata.var_names)
    mat.to_csv(os.path.join(out_dir, "mat.csv"))

    # Export Metadata
    print("    Exporting Metadata...")
    meta = pd.DataFrame(data=adata.obs)
    meta.to_csv(os.path.join(out_dir, "metadata.tsv"), sep="\t")

    # Export Spatial Coordinates
    print("    Exporting Coordinates...")
    # Ensure keys exist
    if 'spatial' in adata.obsm:
        cord = pd.DataFrame(data=adata.obsm['spatial'], index=adata.obs_names, columns=['x', 'y'])
        cord.to_csv(os.path.join(out_dir, "position_spatial.tsv"), sep="\t")

    # Export UMAP Coordinates
    if 'X_umap' in adata.obsm:
        umap = pd.DataFrame(data=adata.obsm["X_umap"], index=adata.obs_names, columns=['x', 'y'])
        umap.to_csv(os.path.join(out_dir, "position_X_umap.tsv"), sep="\t")


adata_info(adata, WORK_DIR)
print(f"    Export completed to: {WORK_DIR}")
print("=" * 60)
print("Pipeline Completed Successfully.")

print("=" * 60)

