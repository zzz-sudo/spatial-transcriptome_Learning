# ===========================================================================
# Project: Spatial Transcriptomics - RNA Velocity Analysis (Embedded SIRV)
# File Name: 35_run_spatial_velocity_sirv_fixed.py
# Author: Kuroneko (PhD Candidate)
# Current Path: F:\ST\code
#
# ---------------------------------------------------------------------------
# [1. 脚本功能描述 (Script Functionality)]
# ---------------------------------------------------------------------------
# 本脚本旨在解决第三方 SIRV 库兼容性问题，通过内置算法实现空间 RNA 速率分析：
# 1. [核心算法]: 内置 SIRV (Spatially Inferred RNA Velocity) 逻辑，不依赖外部包。
#    利用 KNN 算法将 scRNA-seq 的剪接动力学 (Spliced/Unspliced) 映射到空间细胞。
# 2. [数据集成]: 自动处理基因名大小写不一致问题 (如 Malat1 vs MALAT1)，
#    并通过相关性距离 (Correlation Metric) 寻找最佳单细胞邻居。
# 3. [速率流场]: 使用 scVelo 计算 RNA 速率，并将流线图 (Streamline) 投影回
#    真实的物理空间坐标 (xy_loc)，展示组织内的细胞分化轨迹。
#
# ---------------------------------------------------------------------------
# [2. 原始数据来源 (Data Acquisition and Origin)]
# ---------------------------------------------------------------------------
# - 空间数据 (HybISS - Mouse Brain):
#   源自 HybISS 原位测序技术。该技术具有高空间分辨率，但通常无法有效捕获
#   内含子 (Intron) 序列，因此缺失计算 RNA Velocity 所需的 Unspliced 矩阵。
# - 单细胞数据 (scRNA-seq - Mouse Brain):
#   作为参考数据集 (Reference)。包含完整的 Spliced 和 Unspliced 计数矩阵，
#   用于训练模型并向空间数据提供动力学先验信息。
#
# ---------------------------------------------------------------------------
# [3. 输入文件清单 (Input Files Mapping)]
# ---------------------------------------------------------------------------
# - 空间数据: F:\ST\code\data\Mouse_brain\HybISS_adata.h5ad
# - 单细胞数据: F:\ST\code\data\Mouse_brain\RNA_adata.h5ad
#
# ---------------------------------------------------------------------------
# [4. 输出文件清单 (Output Files Mapping)]
# ---------------------------------------------------------------------------
# - 速率结果对象: F:\ST\code\results\sirv_velocity\HybISS_velocity_final.h5ad
# - 空间流场图: F:\ST\code\results\sirv_velocity\spatial_velocity_stream.png
# - UMAP流场图: F:\ST\code\results\sirv_velocity\umap_velocity_stream.png
# ===========================================================================

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from scipy import sparse, stats


# ---------------------------------------------------------------------------
# [Core Algorithm] SIRV 函数定义 (保持在主模块外)
# ---------------------------------------------------------------------------
def SIRV(spatial_adata, sc_adata, k, columns):
    """
    Spatially Inferred RNA Velocity (Internal Implementation)
    """
    print(f"\n[SIRV-Internal] Starting integration with k={k}...")

    # 1. 基因对齐
    spatial_genes = [g.capitalize() for g in spatial_adata.var_names]
    sc_genes = [g.capitalize() for g in sc_adata.var_names]
    spatial_adata.var_names = spatial_genes
    sc_adata.var_names = sc_genes

    common_genes = np.intersect1d(spatial_adata.var_names, sc_adata.var_names)
    print(f"    Found {len(common_genes)} common genes.")

    if len(common_genes) < 50:
        raise ValueError("Critical Error: Too few common genes (<50).")

    # 2. 准备数据
    X_spatial = spatial_adata[:, common_genes].X
    X_sc = sc_adata[:, common_genes].X
    if sparse.issparse(X_spatial): X_spatial = X_spatial.toarray()
    if sparse.issparse(X_sc): X_sc = X_sc.toarray()

    # 3. KNN 搜索
    print("    Calculating Nearest Neighbors...")
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='correlation').fit(X_sc)
    distances, indices = nbrs.kneighbors(X_spatial)

    # 4. 插补
    print("    Imputing velocity layers...")
    if 'spliced' not in sc_adata.layers or 'unspliced' not in sc_adata.layers:
        raise ValueError("Reference scRNA-seq missing spliced/unspliced layers!")

    sc_gene_indices = sc_adata.var_names.get_indexer(common_genes)
    S_sc_subset = sc_adata.layers['spliced'][:, sc_gene_indices]
    U_sc_subset = sc_adata.layers['unspliced'][:, sc_gene_indices]

    # 矩阵化插补
    n_obs = spatial_adata.n_obs
    row_idx = np.repeat(np.arange(n_obs), k)
    col_idx = indices.flatten()
    data = np.ones(len(row_idx)) / k
    W = sparse.csr_matrix((data, (row_idx, col_idx)), shape=(n_obs, sc_adata.n_obs))

    adata_imputed = spatial_adata[:, common_genes].copy()

    print("    Performing matrix multiplication...")
    if sparse.issparse(S_sc_subset):
        S_imp = W.dot(S_sc_subset)
        U_imp = W.dot(U_sc_subset)
    else:
        S_imp = W.dot(sparse.csr_matrix(S_sc_subset))
        U_imp = W.dot(sparse.csr_matrix(U_sc_subset))

    adata_imputed.layers['spliced'] = S_imp
    adata_imputed.layers['unspliced'] = U_imp

    # 5. 标签转移 (已包含 Scipy 兼容性修复)
    for col in columns:
        if col in sc_adata.obs.columns:
            print(f"    Transferring label: {col}")
            ref_labels = sc_adata.obs[col].values
            unique_classes, encoded_ref = np.unique(ref_labels.astype(str), return_inverse=True)
            neighbor_encoded = encoded_ref[indices]
            mode_encoded, _ = stats.mode(neighbor_encoded, axis=1, keepdims=False)
            mode_labels = unique_classes[mode_encoded.flatten()]
            adata_imputed.obs[col] = mode_labels

    return adata_imputed


# ===========================================================================
# [Main Execution Block] Windows 必须把执行代码放在这里！
# ===========================================================================
if __name__ == "__main__":
    # ---------------------------------------------------------------------------
    # [Module 1] 环境配置
    # ---------------------------------------------------------------------------
    print("[Module 1] Initializing Environment...")

    BASE_DIR = r"F:\ST\code"
    DATA_DIR = os.path.join(BASE_DIR, "data", "Mouse_brain")
    OUT_DIR = os.path.join(BASE_DIR, "results", "sirv_velocity")

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    scv.settings.set_figure_params('scvelo', dpi=120, dpi_save=300, transparent=True)
    scv.settings.figdir = OUT_DIR

    # ---------------------------------------------------------------------------
    # [Module 2] 数据加载
    # ---------------------------------------------------------------------------
    print("\n[Module 2] Loading Datasets...")

    spatial_path = os.path.join(DATA_DIR, "HybISS_adata.h5ad")
    sc_path = os.path.join(DATA_DIR, "RNA_adata.h5ad")

    if not os.path.exists(spatial_path) or not os.path.exists(sc_path):
        print("Error: Files not found.")
        sys.exit(1)

    print("    Loading Spatial data...")
    adata_spatial = sc.read(spatial_path)

    print("    Loading Reference data...")
    adata_sc = sc.read(sc_path)

    # ---------------------------------------------------------------------------
    # [Module 3] 运行 SIRV
    # ---------------------------------------------------------------------------
    print("\n[Module 3] Running Internal SIRV Integration...")

    target_cols = ['Region', 'Subclass', 'cell_type', 'Class']
    valid_cols = [c for c in target_cols if c in adata_sc.obs.columns]

    adata_imputed = SIRV(adata_spatial, adata_sc, k=50, columns=valid_cols)

    print("    Restoring original spatial expression counts...")
    original_X = adata_spatial[:, adata_imputed.var_names].X
    if sparse.issparse(original_X):
        adata_imputed.X = original_X
    else:
        adata_imputed.X = sparse.csr_matrix(original_X)

    # ---------------------------------------------------------------------------
    # [Module 4] 预处理与速率计算
    # ---------------------------------------------------------------------------
    print("\n[Module 4] Calculating Velocity...")

    # 1. 归一化 (使用 Scanpy 的 log1p 避免 scVelo 警告)
    scv.pp.normalize_per_cell(adata_imputed, enforce=True)
    sc.pp.log1p(adata_imputed)

    # 2. 基础降维与聚类
    sc.pp.pca(adata_imputed)

    # [Fix] 显式计算邻居，避免 scVelo 自动计算的警告
    print("    Computing neighbors (Scanpy)...")
    sc.pp.neighbors(adata_imputed, n_neighbors=30, n_pcs=30)

    sc.tl.umap(adata_imputed)
    sc.tl.leiden(adata_imputed)

    # 3. 速率计算
    # scVelo 需要计算一阶矩(Ms)和二阶矩(Mu)
    print("    Computing moments...")
    scv.pp.moments(adata_imputed, n_pcs=30, n_neighbors=30)

    print("    Computing velocity (stochastic)...")
    scv.tl.velocity(adata_imputed, mode='stochastic')

    print("    Computing velocity graph (Parallel)...")
    # 这一步是导致 Windows 报错的关键，现在放在 if __name__ 里就安全了
    # n_jobs=-1 表示使用所有 CPU 核心
    scv.tl.velocity_graph(adata_imputed, n_jobs=-1)

    # ---------------------------------------------------------------------------
    # [Module 5] 绘图
    # ---------------------------------------------------------------------------
    print("\n[Module 5] Plotting...")

    spatial_key = None
    for key in ['xy_loc', 'spatial']:
        if key in adata_imputed.obsm:
            spatial_key = key
            break

    if spatial_key:
        print(f"    Plotting spatial velocity using key: {spatial_key}")
        scv.pl.velocity_embedding_stream(
            adata_imputed,
            basis=spatial_key,
            color='leiden',
            size=60,
            title='Spatial RNA Velocity',
            save="spatial_velocity_stream.png"
        )

    scv.pl.velocity_embedding_stream(
        adata_imputed,
        basis='umap',
        color='leiden',
        title='UMAP RNA Velocity',
        save="umap_velocity_stream.png"
    )

    # 保存
    save_path = os.path.join(OUT_DIR, "HybISS_velocity_final.h5ad")
    adata_imputed.write_h5ad(save_path)

    print(f"\nAnalysis Completed. Results saved to: {OUT_DIR}")
