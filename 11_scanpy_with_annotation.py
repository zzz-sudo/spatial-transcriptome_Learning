# -*- coding: utf-8 -*-
"""
===========================================================================
File Name: step1_scanpy_with_annotation.py
Author: Kuroneko
Date: 2026-02-02
Version: V2.0 (With Single-Cell Annotation Integration)

Description:
    [Scanpy Spatial Analysis with SC Reference]
    This script extends the basic workflow by integrating Single-Cell data.

    Changes from V1.0:
    1. Added loading of Single-Cell reference data (.h5ad).
    2. Performed Label Transfer (sc.tl.ingest) to project cell types
       from Single-Cell to Spatial data.
    3. The output metadata now contains 'predicted_celltype'.

Input Files:
    - Spatial: F:/ST/GBM1_spaceranger_out/
    - SingleCell: F:/ST/data/sc_reference.h5ad (Hypothetical path)

Output Files:
    - mat.csv, metadata.tsv (Now includes cell types), position_*.tsv


# Set identity to Python clusters
if ("clusters" %in% colnames(obj@meta.data)) {
  Idents(obj) <- "clusters"
}

# Set identity to Predicted Cell Types
# 如果 Python 生成了 'cell_type' 列，我们就用它
if ("cell_type" %in% colnames(obj@meta.data)) {
  Idents(obj) <- "cell_type"
  cat("    Identity set to 'cell_type' (from Single-Cell Annotation).\n")
} else if ("clusters" %in% colnames(obj@meta.data)) {
  Idents(obj) <- "clusters"
  cat("    Identity set to 'clusters' (Unsupervised).\n")

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
DATA_DIR = r"F:\ST\code\results\scanpy_seurat"
# 假设单细胞数据的路径 (如果没有这个文件，脚本会跳过注释部分)
SC_FILE = r"F:\ST\data\sc_reference.h5ad"
WORK_DIR = r"F:\ST\code\results\scanpy_seurat"

if not os.path.exists(WORK_DIR):
    os.makedirs(WORK_DIR)

sc.settings.verbosity = 3
sc.settings.figdir = WORK_DIR
sc.set_figure_params(dpi=150, facecolor="white", vector_friendly=True)

# 2. Load Visium Data
print("\n[Module 2] Loading Visium Data...")
if not os.path.exists(DATA_DIR):
    print(f"[Error] Data directory not found: {DATA_DIR}")
    exit(1)

adata = sc.read_visium(DATA_DIR)
adata.var_names_make_unique()
print(f"    Visium Shape: {adata.shape}")

# 3. Quality Control (Standard)
print("\n[Module 3] Quality Control...")
adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
sc.pp.filter_cells(adata, min_counts=5000)
sc.pp.filter_cells(adata, max_counts=35000)
adata = adata[adata.obs["pct_counts_mt"] < 20].copy()
sc.pp.filter_genes(adata, min_cells=10)

# 4. Processing
print("\n[Module 4] Processing...")
sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)
sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
# 依然保留 Leiden 聚类作为参考
sc.tl.leiden(adata, key_added="clusters", directed=False, n_iterations=2)

# =========================================================================
# [Module 4.5] Single-Cell Annotation Transfer (核心修改部分)
# =========================================================================
print("\n[Module 4.5] Integration with Single-Cell Data...")

if os.path.exists(SC_FILE):
    print(f"    Loading Single-Cell Reference: {SC_FILE}")
    adata_sc = sc.read_h5ad(SC_FILE)

    # 假设单细胞数据中有一列叫 'cell_type' 存储了注释信息
    # 必须确保单细胞数据的 var_names (基因名) 和空转数据一致

    print("    Processing Reference Data...")
    # 确保单细胞数据也做了相同的预处理
    # sc.pp.normalize_total(adata_sc, target_sum=1e4)
    # sc.pp.log1p(adata_sc)
    # sc.pp.pca(adata_sc)
    # sc.pp.neighbors(adata_sc)

    print("    Running Ingest (Label Transfer)...")
    # 'ingest' 将根据 PCA 空间把 sc 的标签投影给 st
    # obs='cell_type' 是单细胞数据中的注释列名
    sc.tl.ingest(adata, adata_sc, obs='cell_type')

    # 此时 adata.obs['cell_type'] 已经有了预测的细胞类型
    print("    Annotation Transfer Complete. New column 'cell_type' added.")

    # 画图验证
    sc.pl.spatial(adata, img_key="hires", color=["cell_type"], size=1.5, show=False, save="_spatial_annotated.png")

else:
    print("    [Warning] Single-Cell file not found. Skipping annotation transfer.")
    print("    Continuing with Leiden clusters only.")
    # 为了保证后续流程不报错，如果没单细胞，我们就把 clusters 复制一份当做 cell_type
    adata.obs['cell_type'] = adata.obs['clusters']

# =========================================================================

# 5. Export for R
print("\n[Module 6] Exporting Data for Seurat...")


def huage_adata_info(adata, out_dir):
    mat = pd.DataFrame(data=adata.X.todense(), index=adata.obs_names, columns=adata.var_names)
    mat.to_csv(os.path.join(out_dir, "mat.csv"))

    # 关键：这里导出的 metadata.tsv 现在包含了 'cell_type' 列
    meta = pd.DataFrame(data=adata.obs)
    meta.to_csv(os.path.join(out_dir, "metadata.tsv"), sep="\t")

    if 'spatial' in adata.obsm:
        cord = pd.DataFrame(data=adata.obsm['spatial'], index=adata.obs_names, columns=['x', 'y'])
        cord.to_csv(os.path.join(out_dir, "position_spatial.tsv"), sep="\t")


huage_adata_info(adata, WORK_DIR)
print(f"    Export completed to: {WORK_DIR}")
print("=" * 60)
print("Pipeline Completed.")
print("=" * 60)