# -*- coding: utf-8 -*-
"""
===========================================================================
文件名称: st_k8_lymph_node_analysis_v2.py
作    者: Kuroneko
创建日期: 2026-02-01
版    本: V2.0 (Fix tarfile error)

功能描述:
    [10x Visium 空间转录组基础标准分析流程]
    基于 V1_Human_Lymph_Node (人淋巴结) 数据集进行全流程演示。

    修复说明:
    - [核心修复] 添加了 tarfile.data_filter 补丁，解决 Python < 3.11.4 版本
      下 Scanpy 下载数据时的 AttributeError 报错。

    核心步骤:
    1. 数据获取: 自动下载并加载 10x Visium 数据。
    2. 质量控制 (QC): 计算线粒体比例，绘制 QC 小提琴图。
    3. 数据过滤: 剔除低质量斑点。
    4. 降维聚类: 标准化 -> Log1p -> PCA -> UMAP -> Leiden。
    5. 空间可视化: 绘制空间切片图。
    6. 差异分析: 寻找 Marker 基因。

输入文件 (Input):
    - sc.datasets.visium_sge("V1_Human_Lymph_Node") (自动下载)

输出文件 (Output):
    - F:/ST/code/02/V1_Human_Lymph_Node.h5ad
    - F:/ST/code/results/figures_lymph/
===========================================================================
"""

import os
import tarfile  # [修复] 导入 tarfile 模块
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================================
# [Module 0] 环境初始化与补丁修复
# =========================================================================
print("[Module 0] 初始化系统环境...")

# [关键修复] 注入 tarfile.data_filter 补丁
# 解决 AttributeError: module 'tarfile' has no attribute 'data_filter'
if not hasattr(tarfile, "data_filter"):
    print("    [Patch] 检测到旧版 tarfile，正在应用兼容性补丁...")


    def data_filter(member, dest_path):
        return member


    tarfile.data_filter = data_filter

BASE_DIR = r"F:\ST\code"
OUTPUT_FILE = os.path.join(BASE_DIR, "02", "V1_Human_Lymph_Node.h5ad")
FIGURE_DIR = os.path.join(BASE_DIR, "results", "figures_lymph")

if not os.path.exists(FIGURE_DIR):
    os.makedirs(FIGURE_DIR)

sc.settings.verbosity = 3
sc.settings.figdir = FIGURE_DIR
sc.set_figure_params(dpi=150, facecolor="white", vector_friendly=True)

print(f"图表保存目录: {FIGURE_DIR}")
print(f"最终数据输出: {OUTPUT_FILE}")

# =========================================================================
# [Module A] 数据加载与基础质控
# =========================================================================
print("\n[Module A] 加载 V1_Human_Lymph_Node 数据...")

try:
    # 尝试下载数据
    adata = sc.datasets.visium_sge(sample_id="V1_Human_Lymph_Node")
except Exception as e:
    print(f"[Error] 下载失败: {e}")
    print("建议检查网络，或尝试更新 Python 版本。")
    exit()

adata.var_names_make_unique()
print(f"    原始数据规模: {adata.shape}")

# 1. 计算线粒体指标
print("--> 计算质控指标...")
adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

# 2. 绘制过滤前的 QC 分布图
print("--> 绘制 QC 分布直方图...")
fig, axs = plt.subplots(1, 4, figsize=(15, 4))
sns.histplot(adata.obs["total_counts"], kde=False, ax=axs[0])
axs[0].set_title("Total Counts")

sns.histplot(adata.obs["total_counts"][adata.obs["total_counts"] < 10000],
             kde=False, bins=40, ax=axs[1])
axs[1].set_title("Total Counts (<10k)")

sns.histplot(adata.obs["n_genes_by_counts"], kde=False, bins=60, ax=axs[2])
axs[2].set_title("N Genes")

sns.histplot(adata.obs["n_genes_by_counts"][adata.obs["n_genes_by_counts"] < 4000],
             kde=False, bins=60, ax=axs[3])
axs[3].set_title("N Genes (<4k)")

plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "QC_Histograms.png"))
plt.close()

# 3. 绘制空间 QC 图 (过滤前)
sc.pl.spatial(adata, img_key="hires", color=["total_counts", "n_genes_by_counts", "pct_counts_mt"],
              title=["Total Counts", "N Genes", "MT %"],
              show=False, save="_QC_Spatial_Before_Filter.png")

# =========================================================================
# [Module B] 数据过滤与预处理
# =========================================================================
print("\n[Module B] 执行数据过滤与标准化...")

# 1. 过滤低质量细胞
print(f"    过滤前斑点数: {adata.n_obs}")
sc.pp.filter_cells(adata, min_counts=5000)
sc.pp.filter_cells(adata, max_counts=35000)
adata = adata[adata.obs["pct_counts_mt"] < 20].copy()
print(f"    过滤后斑点数: {adata.n_obs}")

# 2. 过滤低表达基因
sc.pp.filter_genes(adata, min_cells=10)

# 3. 归一化与对数变换
print("--> 执行归一化与 Log1p...")
sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)

# 4. 高变基因选择
print("--> 计算高变基因...")
sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)

# =========================================================================
# [Module C] 降维与聚类
# =========================================================================
print("\n[Module C] 降维与 Leiden 聚类...")

# 1. PCA
sc.pp.pca(adata)

# 2. 计算邻居图
sc.pp.neighbors(adata)

# 3. UMAP
sc.tl.umap(adata)

# 4. Leiden 聚类
sc.tl.leiden(adata, key_added="clusters")

# 绘图: UMAP 聚类结果
print("--> 绘制 UMAP...")
sc.pl.umap(adata, color=["total_counts", "n_genes_by_counts", "clusters"],
           wspace=0.4, show=False, save="_UMAP_Analysis.png")

# =========================================================================
# [Module D] 空间可视化与差异分析
# =========================================================================
print("\n[Module D] 空间可视化与 Marker 基因分析...")

# 1. 空间聚类分布图
print("--> 绘制空间聚类分布图...")
sc.pl.spatial(adata, img_key="hires", color="clusters", size=1.5,
              title="Spatial Clusters", show=False, save="_Spatial_Clusters.png")

# 2. 局部放大图 (Crop)
print("--> 绘制局部放大图...")
sc.pl.spatial(adata, img_key="hires", color="clusters",
              groups=["0", "4"], crop_coord=[2000, 6000, 2000, 6000],
              alpha=0.5, size=1.3, title="Zoomed Spatial Clusters",
              show=False, save="_Spatial_Clusters_Zoomed.png")

# 3. 差异基因分析 (Rank Genes Groups)
print("--> 计算 Marker 基因 (t-test)...")
sc.tl.rank_genes_groups(adata, "clusters", method="t-test")

# 4. 绘制热图
print("--> 绘制 Marker 基因热图...")
# 检查 cluster 是否存在再画图
unique_clusters = adata.obs['clusters'].unique().tolist()
if "4" in unique_clusters:
    sc.pl.rank_genes_groups_heatmap(adata, groups="4", n_genes=10, groupby="clusters",
                                    show_gene_labels=True, show=False, save="_Heatmap_Cluster4.png")
if "0" in unique_clusters:
    sc.pl.rank_genes_groups_heatmap(adata, groups="0", n_genes=10, groupby="clusters",
                                    show_gene_labels=True, show=False, save="_Heatmap_Cluster0.png")

# 5. 绘制特定 Marker 基因的空间分布 (CR2)
if "CR2" in adata.var_names:
    print("--> 绘制 CR2 基因空间分布...")
    sc.pl.spatial(adata, img_key="hires", color=["clusters", "CR2"], alpha=0.7,
                  show=False, save="_Spatial_Gene_CR2.png")

# =========================================================================
# [Module E] 结果保存
# =========================================================================
print("\n[Module E] 保存最终结果...")

output_dir = os.path.dirname(OUTPUT_FILE)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

adata.write(OUTPUT_FILE, compression="gzip")

print("-" * 60)
print("V1_Human_Lymph_Node 基础分析流程执行完毕 (V2.0)。")
print(f"数据已保存至: {OUTPUT_FILE}")
print("-" * 60)