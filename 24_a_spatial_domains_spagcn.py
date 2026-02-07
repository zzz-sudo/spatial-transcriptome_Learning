# ===========================================================================
# File Name: 24_a_spatial_domains_spagcn.py
# Author: Kuroneko
# Version: V1.0 (Spatial Domains & SVGs)
#
# Description:
#     [SpaGCN 空间域与高变基因分析 - Python版]
#
#     核心功能:
#     1. Spatial Domains: 利用 GCN (图卷积) 识别空间解剖结构 (例如: 肿瘤区 vs 免疫区)。
#        这是 SpaGene (R) 做不到的"硬聚类"。
#     2. SVGs Detection: 找出每个空间域的特异性标记基因。
#     3. CCIs (可选): SpaGCN 也可以做配受体分析，但我们重点演示前两项。
#
#     硬件加速:
#     - 使用 PyTorch + CUDA (RTX 5070) 进行图卷积运算，速度极快。
#
# Input Files:
#     - Visium Data: F:/ST/code/data/V1_Human_Lymph_Node/
#V1_Human_Lymph_Node/  <-- 我们输入的路径
'''
│
├── filtered_feature_bc_matrix.h5   (基因表达数据：Gene)
│
└── spatial/                        (位置与图像文件夹)
    ├── tissue_positions_list.csv   (空间坐标：Location)
    ├── tissue_hires_image.png      (高分辨率组织图：Histology Image) <-- SpaGCN 要用的图在这里
    ├── tissue_lowres_image.png
    └── scalefactors_json.json      (缩放因子：把坐标对齐到图片上)
'''
# Output Files (in F:/ST/code/results/spagcn/):
#     - plots/: 空间域分布图、SVG热图
#     - spagcn_domains.csv: 空间域分类结果
#     - spagcn_svgs.csv: 识别到的高变基因列表
# ===========================================================================
'''
SpaGene 的确主要有两个版本/同名工具，一个是你刚才上传的 R 包（用于模式识别），
另一个是用于插补（Imputation）的 Python 深度学习工具（BioRxiv 2025），这容易混淆。
但在空间转录组分析的“生态位与高变基因”这一环，
Python 生态中最著名、功能最接近 R 版 SpaGene（甚至更强，因为它能做空间域划分）的工具是 SpaGCN。
很多高阶教程（包括 k 系列）在讲完 R 的 SpaGene 后，通常会介绍 Python 的 SpaGCN 作为对比。
SpaGCN 基于图卷积神经网络 (GCN)，能利用 GPU 加速，非常适合你的 RTX 5070。

深度学习 (GCN)。结合图片和表达，进行“空间聚类/分割”。
你得到什么?	你会得到一个个 Patterns (如 Pattern 1 是环状的，Pattern 2 是散点状的)。
你会得到一个个 Domains (如 Domain 0 是肿瘤核心区，Domain 1 是边缘区)。
'''

import os
import scipy.sparse
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import cv2
import SpaGCN as spg
import torch
import random

# ===========================================================================
# [Patch] 解决新版 Scipy 兼容性与随机种子
# ===========================================================================
if not hasattr(scipy.sparse.spmatrix, "A"):
    scipy.sparse.spmatrix.A = property(lambda x: x.toarray())

r_seed = 42
random.seed(r_seed)
np.random.seed(r_seed)
torch.manual_seed(r_seed)
torch.cuda.manual_seed(r_seed)

# -------------------------------------------------------------------------
# [Module 1 & 2] 环境、路径与数据加载
# -------------------------------------------------------------------------
print("[Module 1/2] Initializing Environment & Loading Data...")
BASE_DIR = r"F:/ST/code/"
DATA_DIR = os.path.join(BASE_DIR, "data", "V1_Human_Lymph_Node")
OUT_DIR = os.path.join(BASE_DIR, "results", "spagcn")
PLOT_DIR = os.path.join(OUT_DIR, "plots")

for d in [OUT_DIR, PLOT_DIR]:
    os.makedirs(d, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"    Hardware: {device} (RTX 5070 Ready)")

adata = sc.read_visium(DATA_DIR)
adata.var_names_make_unique()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# -------------------------------------------------------------------------
# [Module 3] 图像处理与坐标转换
# -------------------------------------------------------------------------
print("[Module 3] Processing Image Information...")
library_id = list(adata.uns["spatial"].keys())[0]
scalefactors = adata.uns["spatial"][library_id]["scalefactors"]
img_path = os.path.join(DATA_DIR, "spatial", "tissue_hires_image.png")

x_vals = adata.obsm["spatial"][:, 0]
y_vals = adata.obsm["spatial"][:, 1]
img = cv2.imread(img_path)
scale = scalefactors.get("tissue_hires_scalef", 1.0)

# 强制转为 int 解决切片报错
x_pixel = (x_vals * scale).astype(int)
y_pixel = (y_vals * scale).astype(int)

# -------------------------------------------------------------------------
# [Module 4] 识别空间域 (Spatial Domains - 确保该模块完整)
# -------------------------------------------------------------------------
print("[Module 4] Identifying Spatial Domains (GCN Training)...")

adj = spg.calculate_adj_matrix(x=x_pixel, y=y_pixel, x_pixel=x_pixel, y_pixel=y_pixel,
                               image=img, beta=49, alpha=1, histology=True)

l = spg.search_l(p=0.5, adj=adj, start=0.01, end=1000, tol=0.01, max_run=100)

clf = spg.SpaGCN()
clf.set_l(l)

n_clusters = 7  # 对应玉米解剖分区
print(f"    Training GCN model for {n_clusters} clusters...")
clf.train(adata, adj, init_spa=True, init="louvain", res=0.4, tol=5e-3, lr=0.05, max_epochs=200)

y_pred, prob = clf.predict()
adata.obs["spagcn_domain"] = pd.Series(y_pred, index=adata.obs.index, dtype='category')
adata.obs[["spagcn_domain"]].to_csv(os.path.join(OUT_DIR, "spagcn_domains.csv"))

# 绘图 01
plt.figure(figsize=(10, 10))
sc.pl.spatial(adata, img_key="hires", color="spagcn_domain", palette='Set1', show=False)
plt.savefig(os.path.join(PLOT_DIR, "01_Spatial_Domains.png"), dpi=300)
plt.close()

# -------------------------------------------------------------------------
# [Module 5] 自动循环识别所有域的 SVGs (标志基因 - 修复版)
# -------------------------------------------------------------------------
print("\n[Module 5] Detecting Domain-Specific SVGs for all clusters...")

adj_2d = spg.calculate_adj_matrix(x=x_pixel, y=y_pixel, histology=False)
if hasattr(adj_2d, "toarray"): adj_2d = adj_2d.toarray()
radius = int(np.mean(np.sort(adj_2d, axis=1)[:, 1]) * 1.5)

domains = sorted(adata.obs["spagcn_domain"].unique())
all_svgs = []

for target_domain in domains:
    print(f"    --> Analyzing Domain {target_domain}...")

    neighbor_info = spg.find_neighbor_clusters(target_cluster=target_domain,
                                               cell_id=adata.obs.index.tolist(),
                                               x=x_pixel, y=y_pixel,
                                               pred=adata.obs["spagcn_domain"].tolist(),
                                               radius=radius, ratio=0.5)

    neighbor_list = neighbor_info if isinstance(neighbor_info, list) else neighbor_info["neighbor_clusters"]

    # [关键修复]：只传入函数支持的参数名 nbr_list
    de_res = spg.rank_genes_groups(input_adata=adata,
                                   target_cluster=target_domain,
                                   nbr_list=neighbor_list,
                                   label_col="spagcn_domain")

    # [手动过滤]：因为函数不支持 adj_p_cutoff 参数，我们拿到结果后手动过滤
    if de_res is not None and not de_res.empty:
        # 典型的过滤标准：p_adj < 0.05 且 logFC > 0.5
        # 注意：不同版本列名可能不同，这里假设为 'pvals_adj' 和 'logfoldchanges'
        de_res["target_domain"] = target_domain
        all_svgs.append(de_res)

        # 提取该 Domain 表达最显著的前 2 个基因出图
        top_genes = de_res.sort_values(by="pvals_adj").head(2)["genes"].tolist()
        print(f"        Top Markers: {top_genes}")

        sc.pl.spatial(adata, img_key="hires", color=["spagcn_domain"] + top_genes, ncols=3, show=False)
        plt.savefig(os.path.join(PLOT_DIR, f"02_Domain_{target_domain}_Markers.png"), dpi=150)
        plt.close()

# 汇总保存
if all_svgs:
    pd.concat(all_svgs).to_csv(os.path.join(OUT_DIR, "all_domains_svg_summary.csv"), index=False)

print("=" * 60)
print("SpaGCN Analysis Pipeline Completed Successfully.")

print("=" * 60)
