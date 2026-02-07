# ===========================================================================
# File Name: 17_run_deconvolution_c2l.py
# Author: Kuroneko
# Version: V1.0 (The Foundation)
#
# Description:
#     [Cell2location 去卷积 - 核心步骤]
#
#     功能:
#     1. 参照模型训练: 从 scRNA-seq 中提取细胞类型的表达指纹 (Signatures)。
#     2. 空间映射: 将指纹映射到 Visium Spots，计算每个 Spot 中各细胞类型的绝对丰度。
#     3. 导出: 生成 CSV 和 h5ad，作为 CellChat/MISTy 的精确输入。
#
#     硬件优化:
#     - 针对 RTX 5070 启用 CUDA 加速。
#     - 显存管理: 使用 batch_size 防止 OOM。
#
# Input Files:
#     - Spatial: F:/ST/code/data/V1_Human_Lymph_Node/
#     - Single Cell: F:/ST/code/03/sc.h5ad
#
# Output Files (in F:/ST/code/results/deconvolution/):
#     - c2l_results.h5ad: 包含去卷积结果的空间对象
#     - cell_abundance.csv: 细胞丰度矩阵 (供 R 使用)
#     - plots/: 质控与结果图
# ===========================================================================

import sys
import os
import gc
import scanpy as sc
import anndata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import cell2location

from matplotlib import rcParams

# -------------------------------------------------------------------------
# [Module 1] 环境与硬件检查
# -------------------------------------------------------------------------
print("[Module 1] Initializing Environment...")

# 路径设置
BASE_DIR = r"F:/ST/code/"
SPATIAL_DIR = os.path.join(BASE_DIR, "data", "V1_Human_Lymph_Node")
SC_FILE = os.path.join(BASE_DIR, "03", "sc.h5ad")
OUT_DIR = os.path.join(BASE_DIR, "results", "deconvolution")
PLOT_DIR = os.path.join(OUT_DIR, "plots")

for d in [OUT_DIR, PLOT_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

# 设置绘图参数
sc.settings.figdir = PLOT_DIR
sc.settings.set_figure_params(dpi=120, frameon=False, facecolor='white')

# 检查 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"    Hardware: {device}")
if device.type == 'cuda':
    print(f"    GPU Model: {torch.cuda.get_device_name(0)}")
else:
    print("    [Warning] Running on CPU! This will be very slow.")
import scvi
# Cell2location 全局设置
scvi.settings.seed = 42
scvi.settings.dtype = "float32"

# -------------------------------------------------------------------------
# [Module 2] 数据加载与预处理
# -------------------------------------------------------------------------
print("\n[Module 2] Loading Data...")

# 1. 加载单细胞数据 (Reference)
print(f"    Loading Single Cell Ref: {SC_FILE}")
adata_sc = sc.read_h5ad(SC_FILE)
adata_sc.var_names_make_unique()

# [关键] Cell2location 需要 Raw Counts (整数)
# 检查 raw 或者 layers['counts']
if 'counts' in adata_sc.layers:
    adata_sc.X = adata_sc.layers['counts'].copy()
elif adata_sc.raw is not None:
    adata_sc.X = adata_sc.raw.X.copy()
else:
    print("    [Warning] Using .X as counts. Ensure it is NOT log-normalized!")

# 过滤低质量基因
sc.pp.filter_genes(adata_sc, min_cells=10)

# 2. 加载空间数据 (Visium)
print(f"    Loading Visium Spatial: {SPATIAL_DIR}")
adata_vis = sc.read_visium(SPATIAL_DIR)
adata_vis.var_names_make_unique()
sc.pp.filter_genes(adata_vis, min_cells=10)

# 3. 寻找交集基因 (Intersection)
intersect_genes = np.intersect1d(adata_sc.var_names, adata_vis.var_names)
print(f"    Genes in Reference: {adata_sc.n_vars}")
print(f"    Genes in Spatial: {adata_vis.n_vars}")
print(f"    Intersect Genes: {len(intersect_genes)}")

# 削减数据以节省内存，仅保留交集基因
adata_sc = adata_sc[:, intersect_genes].copy()
adata_vis = adata_vis[:, intersect_genes].copy()

# -------------------------------------------------------------------------
# [Module 3] 训练参考模型 (Reference Signature)
# -------------------------------------------------------------------------
print("\n[Module 3] Training Reference Model (NB Regression)...")

# 准备参考模型
# batch_key: 如果你有多个单细胞样本的批次信息，填这里 (例如 'Sample')
# labels_key: 细胞类型注释列名 (请根据你的 sc.h5ad 修改，通常是 'cell_type' 或 'cluster')
labels_key = 'cell_type'
# 自动检测列名
for col in ['cell_type', 'CellType', 'annotation', 'cluster', 'leiden']:
    if col in adata_sc.obs.columns:
        labels_key = col
        break
print(f"    Using '{labels_key}' as cell type label.")

cell2location.models.RegressionModel.setup_anndata(
    adata=adata_sc,
    labels_key=labels_key,
    # batch_key="Sample" # 如果有批次效应需开启
)

# 建立模型
mod_ref = cell2location.models.RegressionModel(adata_sc)

# 训练 (RTX 5070 应该很快)
# max_epochs: 250 是参考模型的标准值
print("    Training on GPU...")
mod_ref.train(max_epochs=250, batch_size=2500)

# 导出基因指纹 (Signatures)
# 这一步得到每个细胞类型的基因表达谱
adata_ref = mod_ref.export_posterior(
    adata_sc, sample_kwargs={'num_samples': 1000, 'batch_size': 2500}
)

# 保存指纹图
mod_ref.plot_QC()
plt.savefig(os.path.join(PLOT_DIR, "01_Ref_QC.pdf"))
plt.close()

print(f"    Reference Signatures Extracted. Shape: {adata_ref.varm['means_per_cluster_mu_fg'].shape}")

# 清理显存
del mod_ref, adata_sc
gc.collect()
torch.cuda.empty_cache()

# -------------------------------------------------------------------------
# [Module 4] 空间映射 (Spatial Mapping)
# -------------------------------------------------------------------------
print("\n[Module 4] Mapping to Spatial (Cell2location)...")

# 准备空间数据
cell2location.models.Cell2location.setup_anndata(adata=adata_vis)

# 建立模型
# inf_aver: 单细胞数据得出的平均表达谱
inf_aver = adata_ref.varm['means_per_cluster_mu_fg']

mod_spatial = cell2location.models.Cell2location(
    adata_vis,
    cell_state_df=inf_aver,
    # N_cells_per_location: 这是一个先验知识。
    # Visium 每个点大概有 1-10 个细胞，平均设为 30 (包含多种类型) 是比较宽松的设定
    N_cells_per_location=30,
    detection_alpha=20 # Visium 标准参数
)

# 训练
print("    Training Spatial Model (This is the heavy part)...")
# max_epochs: 空间映射需要更多迭代，通常 500-1000。这里设 1000 以保证收敛。
mod_spatial.train(
    max_epochs=1000,
    batch_size=None,
    train_size=1.0
)

# 导出结果
print("    Exporting posterior distribution...")
adata_vis = mod_spatial.export_posterior(
    adata_vis, sample_kwargs={'num_samples': 1000, 'batch_size': 1000}
)

# 保存 QC 图
mod_spatial.plot_QC()
plt.savefig(os.path.join(PLOT_DIR, "02_Spatial_QC.pdf"))
plt.close()

# -------------------------------------------------------------------------
# [Module 5] 结果提取与保存 (修复版)
# -------------------------------------------------------------------------
print("\n[Module 5] Saving Results...")

# 1. 提取细胞丰度 (q05 是 5% 分位数，比均值更稳健)
# Cell2location 的结果存储在 adata_vis.obsm['q05_cell_abundance_w_sf']
if 'q05_cell_abundance_w_sf' in adata_vis.obsm:
    abundance_df = adata_vis.obsm['q05_cell_abundance_w_sf']
else:
    # 备用方案：如果 q05 不存在，尝试读取 mean
    print("    [Warning] q05 abundance not found, using means...")
    abundance_df = adata_vis.obsm['means_cell_abundance_w_sf']

# 清洗列名:
# 1. 去掉 'q05_cell_abundance_w_sf_' 前缀
abundance_df.columns = [col.replace('q05_cell_abundance_w_sf_', '').replace('means_cell_abundance_w_sf_', '') for col in abundance_df.columns]

# 2. [关键修复] 替换掉非法字符 (如 Treg/Tfr 中的 /)，防止 h5ad 保存失败
print("    Sanitizing column names (removing '/' and '\\')...")
abundance_df.columns = [col.replace('/', '_').replace('\\', '_') for col in abundance_df.columns]

# 将清洗后的结果存回 adata.obsm
adata_vis.obsm['cell_abundance'] = abundance_df

# 2. 保存 CSV (供 R 语言读取)
csv_path = os.path.join(OUT_DIR, "cell_abundance.csv")
abundance_df.to_csv(csv_path)
print(f"    CSV saved: {csv_path}")

# 3. 保存 h5ad (供 Python 下游分析读取)
# 由于列名已经清洗过，这里不会再报错了
h5ad_path = os.path.join(OUT_DIR, "c2l_results.h5ad")
try:
    adata_vis.write_h5ad(h5ad_path)
    print(f"    H5AD saved: {h5ad_path}")
except Exception as e:
    print(f"    [Error] H5AD save failed: {e}")
    print("    Please use the 'cell_abundance.csv' for downstream analysis.")

# 4. 可视化部分结果
print("    Generating spatial plots...")
# 选取丰度最高的前 4 种细胞类型绘图
top_cells = abundance_df.sum().sort_values(ascending=False).head(4).index.tolist()

with rcParams.context({'figure.figsize': (8, 8)}):
    sc.pl.spatial(
        adata_vis,
        color=top_cells,
        layer="counts",
        frameon=False,
        show=False
    )
    plt.savefig(os.path.join(PLOT_DIR, "03_Top_CellTypes_Spatial.pdf"))
    plt.close()

print("=" * 60)
print("Deconvolution Completed Successfully.")

print("=" * 60)
