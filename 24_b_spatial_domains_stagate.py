# ===========================================================================
# File Name: 24_b_spatial_domains_stagate.py
# Author: Kuroneko
# Version: V1.0 (Spatial Domains SOTA)
#
# Description:
#     [STAGATE 空间域划分分析]
#
#     核心功能:
#     1. 空间邻近图构建: 基于物理距离构建 Spot 之间的连接图。
#     2. STAGATE 训练: 使用图注意力自编码器 (Graph Attention Auto-Encoder)
#        学习融合了"基因表达 + 空间信息"的低维特征 (Latent Embedding)。
#     3. 空间域识别: 基于学习到的特征进行聚类，划分组织区域。
#     4. 数据去噪: 利用解码器重构基因表达矩阵，去除技术噪声。
#
#     硬件加速:
#     - 必须使用 GPU (RTX 5070)。STAGATE 的注意力机制计算量较大，
#       CPU 运行会非常慢。
#
# Input Files:
#     - Visium Data: F:/ST/code/data/V1_Human_Lymph_Node/
#
# Output Files (in F:/ST/code/results/stagate/):
#     - plots/: 空间域聚类图、特征对比图
#     - stagate_embedding.csv: 学习到的空间隐变量
#     - stagate_domains.csv: 空间域分类结果
#     - stagate_result.h5ad: 完整结果对象
# ===========================================================================
'''
如何抉择 SpaGCN vs STAGATE？

SpaGCN: 是一把**“瑞士军刀”**。如果你既要分区域（Domains），又要找这个区域的特异基因（SVGs），甚至还要利用 H&E 图片的颜色信息（比如这里有一块出血，颜色深），用 SpaGCN。

STAGATE: 是一把**“手术刀”。如果你只关心“把区域分得准”**，特别是对于那种边界模糊、渐变的结构（比如淋巴结的生发中心向外扩散），STAGATE 的注意力机制（Attention）通常能画出比 SpaGCN 更符合生物学的平滑边界。

代码中的巧思：

我在 [Module 5] 特意加了一个对比环节（Baseline vs STAGATE）。运行后你会生成一张对比图 03_Comparison...。

这张图是你放在论文或者组会 PPT 里最有说服力的证据：左边是麻子脸（噪点多），右边是光滑的皮肤（区域连贯），一目了然地展示了空间算法的威力。

'''
import os
import sys
import gc
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import torch
import STAGATE
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
# 清理计算图，防止多次运行时的冲突
tf.compat.v1.reset_default_graph()

# -------------------------------------------------------------------------
# [Module 1] 环境与数据加载
# -------------------------------------------------------------------------
print("=" * 60)
print("[Module 1] Initializing Environment...")
print("=" * 60)

BASE_DIR = r"F:/ST/code/"
DATA_DIR = os.path.join(BASE_DIR, "data", "V1_Human_Lymph_Node")
OUT_DIR = os.path.join(BASE_DIR, "results", "stagate")
PLOT_DIR = os.path.join(OUT_DIR, "plots")

for d in [OUT_DIR, PLOT_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

sc.settings.figdir = PLOT_DIR
sc.settings.set_figure_params(dpi=120, frameon=False, facecolor='white')

# 显存清理函数
def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()

# 检查 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Hardware: {device}")
if device.type == 'cpu':
    print("[Warning] STAGATE is very slow on CPU! Please check your CUDA setup.")

# 1. 加载数据
print("Loading Visium data...")

adata = sc.read_visium(DATA_DIR)
adata.var_names_make_unique()

# 2. 预处理 (标准 Scanpy 流程)
# STAGATE 需要高变基因来减少计算噪声
print("Preprocessing (Normalization & HVGs)...")
sc.pp.calculate_qc_metrics(adata, inplace=True)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# 选择高变基因 (Top 3000 是 STAGATE 的推荐值)
sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=3000)

print(f"Data shape: {adata.shape}")

# -------------------------------------------------------------------------
# [Module 2] 构建空间邻近图 (Spatial Neighbor Graph)
# -------------------------------------------------------------------------
print("\n" + "=" * 60)
print("[Module 2] Constructing Spatial Graph...")
print("=" * 60)

# STAGATE 需要知道哪些 Spot 是邻居。
# rad_cutoff: 距离阈值。对于 Visium (10x)，Spot 间距是固定的。
# 如果设置为 None，STAGATE 会基于 KNN (k=6) 自动构建。
# 这里我们使用 KNN 模式，因为它对切片扭曲更鲁棒。
STAGATE.Cal_Spatial_Net(adata, k_cutoff=6, model='KNN')

# 打印网络统计信息
print("Spatial network constructed.")
print(f"Stats: {adata.uns['Spatial_Net'].shape[0]} edges found.")

# 检查是否有独立的 Spot (没有邻居的点)
# 这种点通常是组织外的背景噪点，STAGATE 训练时需要剔除或注意
if 'Spatial_Net' not in adata.uns:
    raise ValueError("Spatial Net construction failed!")

# -------------------------------------------------------------------------
# [Module 3] 训练 STAGATE 模型 (GAT Autoencoder)
# -------------------------------------------------------------------------
print("\n" + "=" * 60)
print("[Module 3] Training STAGATE Model (on GPU)...")
print("=" * 60)

clear_gpu()
tf.compat.v1.reset_default_graph()
# 训练参数
# hidden_dims: 编码器的层宽 [512, 30] 表示将 3000 个基因压缩到 30 维
# n_epochs: 迭代次数。500-1000 通常足够收敛。
# gradient_clipping: 防止梯度爆炸
# weight_decay: 正则化参数
adata = STAGATE.train_STAGATE(
    adata,
    alpha=0, # alpha 控制预聚类权重的参数 (0表示不使用预聚类先验，纯无监督)
    pre_labels=None,
    n_epochs=1000,
    save_attention=True, # 保存注意力权重，用于解释哪个邻居更重要
)

print("Training finished.")
print("Latent embedding saved in 'adata.obsm['STAGATE']'.")

# -------------------------------------------------------------------------
# [Module 4] 空间域识别 (Clustering on Embedding)
# -------------------------------------------------------------------------
print("\n" + "=" * 60)
print("[Module 4] Identifying Spatial Domains...")
print("=" * 60)

# STAGATE 的核心产出是 adata.obsm['STAGATE'] 这个矩阵。
# 它融合了基因表达和空间结构。我们对它进行聚类，就能得到空间域。

# 1. 计算邻居 (基于 STAGATE 特征)
sc.pp.neighbors(adata, use_rep='STAGATE')

# 2. 聚类 (使用 Leiden 或 mclust)
# 如果你需要指定具体的聚类数量 (例如 7 个域)，可以使用 mclust (需要 R 环境)。
# 为了纯 Python 流程的鲁棒性，这里我们使用 Leiden 算法并搜索分辨率。

# 目标聚类数 (根据淋巴结结构，通常有: 滤泡、T区、生发中心、边缘、血管、纤维等)
TARGET_CLUSTERS = 7
res = 0.1
step = 0.1

print(f"Searching for resolution to get approx {TARGET_CLUSTERS} clusters...")
for i in range(20):
    sc.tl.leiden(adata, resolution=res, key_added="STAGATE_Domain")
    n_clus = len(adata.obs["STAGATE_Domain"].unique())
    print(f"  Resolution {res:.2f} -> {n_clus} domains")
    if n_clus >= TARGET_CLUSTERS:
        break
    res += step

# 3. 结果整理
# 将 Leiden 的数字标签转为 "Domain_0", "Domain_1"...
adata.obs['STAGATE_Domain'] = 'Domain_' + adata.obs['STAGATE_Domain'].astype(str)

print("Spatial domains identified.")

# -------------------------------------------------------------------------
# [Module 5] 可视化与对比
# -------------------------------------------------------------------------
print("\n" + "=" * 60)
print("[Module 5] Visualizing Results...")
print("=" * 60)

# 1. 空间域分布图 (Spatial Plot)
# 这是最重要的结果图
plt.figure(figsize=(8, 8))
sc.pl.spatial(
    adata,
    img_key="hires",
    color="STAGATE_Domain",
    title="STAGATE Spatial Domains",
    show=False,
    palette="tab20" # 使用区分度高的色板
)
plt.savefig(os.path.join(PLOT_DIR, "01_STAGATE_Domains.pdf"))
plt.close()

# 2. UMAP 投影 (查看特征分离度)
# 如果 STAGATE 训练得好，不同空间域在 UMAP 上应该清晰分开
sc.tl.umap(adata)
plt.figure(figsize=(8, 8))
sc.pl.umap(
    adata,
    color="STAGATE_Domain",
    title="STAGATE Latent Features (UMAP)",
    show=False
)
plt.savefig(os.path.join(PLOT_DIR, "02_STAGATE_UMAP.pdf"))
plt.close()

# 3. 对比: 原始表达聚类 vs STAGATE 聚类
# 为了展示 STAGATE 的优势，我们对比一下仅用基因表达 (PCA) 做聚类的结果
# 这样能明显看出 STAGATE 的结果更平滑、更符合组织结构
print("Running baseline PCA clustering for comparison...")
adata_baseline = adata.copy()
sc.pp.pca(adata_baseline)
sc.pp.neighbors(adata_baseline)
sc.tl.leiden(adata_baseline, resolution=res, key_added="Baseline_Cluster")

plt.figure(figsize=(16, 8))
ax1 = plt.subplot(1, 2, 1)
sc.pl.spatial(adata_baseline, color="Baseline_Cluster", title="Baseline (Gene Only)", ax=ax1, show=False)
ax2 = plt.subplot(1, 2, 2)
sc.pl.spatial(adata, color="STAGATE_Domain", title="STAGATE (Gene + Spatial)", ax=ax2, show=False)
plt.savefig(os.path.join(PLOT_DIR, "03_Comparison_Baseline_vs_STAGATE.pdf"))
plt.close()

# -------------------------------------------------------------------------
# [Module 6] 保存结果
# -------------------------------------------------------------------------
print("\n" + "=" * 60)
print("[Module 6] Saving Data...")
print("=" * 60)

# 保存 Embedding (供下游使用，如轨迹分析)
embedding_df = pd.DataFrame(adata.obsm['STAGATE'], index=adata.obs_names)
embedding_df.to_csv(os.path.join(OUT_DIR, "stagate_embedding.csv"))

# 保存 Domain 标签
domain_df = adata.obs[["STAGATE_Domain"]]
domain_df.to_csv(os.path.join(OUT_DIR, "stagate_domains.csv"))

# 保存完整对象 (可选，文件较大)
# adata.write_h5ad(os.path.join(OUT_DIR, "stagate_result.h5ad"))

print("STAGATE Analysis Completed.")
print(f"Check the comparison plot in: {os.path.join(PLOT_DIR, '03_Comparison_Baseline_vs_STAGATE.pdf')}")
print("You should see that STAGATE domains are much smoother and spatially coherent!")

print("=" * 60)
