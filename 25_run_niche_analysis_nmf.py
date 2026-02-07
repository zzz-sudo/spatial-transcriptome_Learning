# ===========================================================================
# File Name: 25_run_niche_analysis_nmf.py
# Author: Kuroneko
# Version: V1.0 (Educational NMF Edition)
#
# Description:
#     [基于 NMF 的空间生态位 (Spatial Niche) 分析]
#
#     功能说明:
#     本脚本旨在利用 Cell2location 的去卷积结果，识别组织中重复出现的"细胞微环境"（即生态位）。
#     方法的核心是使用非负矩阵分解 (NMF) 将复杂的细胞组成数据降维分解为几个典型的"模式"。
#
#     核心输入:
#     - Cell2location 的结果 (h5ad 格式)，其中 obsm['cell_abundance'] 包含了
#       每个 Spot 中各细胞类型的估计丰度。
#
#     核心输出:
#     1. Niche Signatures (生态位特征): 热图展示每个生态位由哪些细胞类型主导。
#     2. Spatial Map (空间分布): 展示这些生态位在组织切片上的分布位置。
#        - "硬"分类图: 每个 Spot 归属于其权重最大的生态位。
#        - "软"分布图: 分别展示每个生态位的权重分布。
#
#     硬件需求:
#     - CPU 计算为主 (scikit-learn NMF)。RTX 5070 的强大 CPU 足以应对。
#
# Input Files:
#     - C2L Results: F:/ST/code/results/deconvolution/c2l_results.h5ad
#
# Output Files (in F:/ST/code/results/niche_analysis/):
#     - plots/: 包含生态位特征热图和空间分布图。
#     - niche_signatures.csv: 生态位与细胞类型的对应关系表。
#     - spot_niche_weights.csv: 每个 Spot 的生态位权重表。
# ===========================================================================

import os
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import NMF

# -------------------------------------------------------------------------
# [Module 1] 数据加载 (联合分析：读取第 22 步的去卷积结果)
# -------------------------------------------------------------------------
print("[Module 1] Loading Deconvolution Results...")

BASE_DIR = r"F:/ST/code/"
C2L_H5AD = os.path.join(BASE_DIR, "results/deconvolution/c2l_results.h5ad")
OUT_DIR = os.path.join(BASE_DIR, "results/niche_analysis")
PLOT_DIR = os.path.join(OUT_DIR, "plots")

os.makedirs(PLOT_DIR, exist_ok=True)

# 加载数据
adata = sc.read_h5ad(C2L_H5AD)

# 提取细胞丰度矩阵 (对应 Module 22 导出的结果)
cell_type_df = adata.obsm['cell_abundance']
print(f"    Data ready: {cell_type_df.shape[0]} spots x {cell_type_df.shape[1]} cell types")

# -------------------------------------------------------------------------
# [Module 2] 运行 NMF 分解 (识别生态位模式)
# -------------------------------------------------------------------------
n_niches = 6
print(f"[Module 2] Decomposing into {n_niches} Spatial Niches via NMF...")

nmf_model = NMF(n_components=n_niches, init='nndsvd', random_state=42, max_iter=1000)
W = nmf_model.fit_transform(cell_type_df)
H = nmf_model.components_

niche_names = [f"Niche_{i}" for i in range(n_niches)]
niche_weights_df = pd.DataFrame(W, index=adata.obs_names, columns=niche_names)

# [核心修复]：必须将权重矩阵合并到 adata.obs 中，scanpy 绘图才能找到列名
# 这样解决了 KeyError: 'Could not find key Niche_0'
adata.obs = pd.concat([adata.obs, niche_weights_df], axis=1)

# 硬分类：每个 Spot 归属于权重最高的一个 Niche
adata.obs['niche_labels'] = niche_weights_df.idxmax(axis=1).astype('category')

# -------------------------------------------------------------------------
# [Module 3] Niche Signatures (生态位特征热图)
# -------------------------------------------------------------------------
print("[Module 3] Plotting Niche Composition...")

signature_df = pd.DataFrame(H, index=niche_names, columns=cell_type_df.columns)

plt.figure(figsize=(12, 6))
sns.heatmap(signature_df, cmap="YlGnBu")
plt.title("Cell Type Contributions to Each Niche")
plt.savefig(os.path.join(PLOT_DIR, "01_Niche_Signatures_Heatmap.pdf"), bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------------
# [Module 4] 空间映射 (联合分析：将生态位映射到玉米组织图)
# -------------------------------------------------------------------------
print("[Module 4] Plotting Spatial Distributions...")

# 1. 硬分类图
sc.pl.spatial(adata, color='niche_labels', img_key="hires",
              title="Spatial Niche Domains (Hard Assignment)", show=False)
plt.savefig(os.path.join(PLOT_DIR, "02_Spatial_Niches_Map.pdf"))
plt.close()

# 2. 软分类图 (现在不会报错了)
sc.pl.spatial(adata, color=niche_names, ncols=3, img_key="hires",
              cmap="magma", show=False)
plt.savefig(os.path.join(PLOT_DIR, "03_Niche_Weights_Faceted.pdf"))
plt.close()

# -------------------------------------------------------------------------
# [Module 5] 结果导出
# -------------------------------------------------------------------------
niche_weights_df.to_csv(os.path.join(OUT_DIR, "spot_niche_weights.csv"))
signature_df.to_csv(os.path.join(OUT_DIR, "niche_signatures.csv"))
adata.write_h5ad(os.path.join(OUT_DIR, "niche_analysis_results.h5ad"))
print("=" * 60)
print("Niche Analysis Finished Successfully.")

print("=" * 60)
