# -*- coding: utf-8 -*-
"""
---------------------------------------------------------------------------
文件名称: st_final_analysis.py
作    者: Kuroneko
功能描述:
    空间转录组(Spatial Transcriptomics)标准分析全流程。
    直接读取 01 文件夹下的原始数据，无需移动文件。
    整合了数据读取、高级质控(Seaborn)、严格过滤、归一化、降维聚类、
    空间可视化(含局部裁剪)、差异分析及数据导出。

输入文件:
    F:/ST/code/01/GSE194329_RAW/GBM1_spaceranger_out
    (需包含 filtered_feature_bc_matrix.h5 和 spatial 文件夹)

输出文件:
    F:/ST/code/results/ (包含所有分析图片、CSV表格、最终H5AD存档)
---------------------------------------------------------------------------
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import sparse
from scipy.io import mmwrite

# ================= 1. 环境与路径设置 =================
print("Step 1: 环境配置与路径检查...")

# 基础工作目录
base_dir = r"F:\ST\code"

# [修正] 输入数据路径: 直接指向 01 文件夹下的具体路径
input_dir = os.path.join(base_dir, r"01\GSE194329_RAW\GBM1_spaceranger_out")

# 输出结果路径
result_dir = os.path.join(base_dir, "results")
figure_dir = os.path.join(result_dir, "figures")
matrix_dir = os.path.join(result_dir, "sparse_matrix")

# 自动创建不存在的输出目录
for path in [result_dir, figure_dir, matrix_dir]:
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"已创建目录: {path}")

# 设置 Scanpy 绘图参数 (分辨率 dpi=150 适合屏幕查看)
sc.settings.verbosity = 3
sc.settings.figdir = figure_dir
sc.set_figure_params(dpi=150, facecolor="white", figsize=(8, 8))

print(f"输入目录: {input_dir}")
print(f"输出目录: {result_dir}")

# ================= 2. 数据读取与基础探索 =================
print("\nStep 2: 读取数据并探索...")

# 读取 10x Visium 数据
try:
    adata = sc.read_visium(input_dir, count_file='filtered_feature_bc_matrix.h5')
    adata.var_names_make_unique()  # 确保基因名唯一
except Exception as e:
    print(f"读取失败! 请检查路径: {input_dir}")
    print(f"错误信息: {e}")
    exit()

# 打印数据基本形状
print(f"数据加载成功。")
print(f"Spots (行数): {adata.n_obs}")
print(f"Genes (列数): {adata.n_vars}")

# 备份原始计数 (这对后续分析非常重要，防止归一化覆盖原始数据)
adata.layers["raw_counts"] = adata.X.copy()
print("已备份原始计数至 adata.layers['raw_counts']")

# ================= 3. 高级质控 (QC) =================
print("\nStep 3: 计算质控指标并绘图...")

# 1. 标记线粒体基因 (MT- 开头)
adata.var["mt"] = adata.var_names.str.startswith("MT-")

# 2. 计算 QC 指标 (total_counts, n_genes_by_counts, pct_counts_mt)
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

# 3. 绘制详细的质控分布图 (使用 Seaborn)
fig, axs = plt.subplots(1, 4, figsize=(16, 4))

# 图1: 总 Counts 分布
sns.histplot(adata.obs["total_counts"], kde=False, ax=axs[0])
axs[0].set_title("Total Counts per Spot")

# 图2: 低 Count 区域细节 (帮助确定 min_counts)
sns.histplot(adata.obs["total_counts"][adata.obs["total_counts"] < 10000], kde=False, bins=40, ax=axs[1])
axs[1].set_title("Total Counts (<10k detail)")

# 图3: 基因数量分布
sns.histplot(adata.obs["n_genes_by_counts"], kde=False, bins=60, ax=axs[2])
axs[2].set_title("Genes per Spot")

# 图4: 线粒体比例分布
sns.histplot(adata.obs["pct_counts_mt"], kde=False, bins=60, ax=axs[3])
axs[3].set_title("Mitochondrial %")

plt.tight_layout()
plt.savefig(os.path.join(figure_dir, "01_qc_histograms_seaborn.png"))
plt.close()
print("质控直方图已保存。")

# 4. 绘制空间质控图 (直接在切片上看质量)
sc.pl.spatial(adata, img_key="hires", color=["total_counts", "pct_counts_mt"],
              show=False, save="_qc_spatial.png")

# ================= 4. 数据过滤 =================
print("\nStep 4: 执行数据过滤...")
print(f"过滤前 Spot 数量: {adata.n_obs}")

# [关键参数] 使用严格参数
# 过滤掉 Counts 小于 5000 或 大于 35000 的 Spot
sc.pp.filter_cells(adata, min_counts=5000)
sc.pp.filter_cells(adata, max_counts=35000)

# 过滤掉线粒体比例大于 20% 的 Spot
adata = adata[adata.obs["pct_counts_mt"] < 20, :].copy()

# 过滤掉在少于 10 个 Spot 中表达的基因
sc.pp.filter_genes(adata, min_cells=10)

print(f"过滤后 Spot 数量: {adata.n_obs}")

# ================= 5. 归一化与对数化 =================
print("\nStep 5: 归一化与对数化...")

# 归一化到每百万 (CPM)
sc.pp.normalize_total(adata, target_sum=1e4)

# 备份归一化后的数据
adata.layers["cpm_counts"] = adata.X.copy()

# 对数化 log(x+1)
sc.pp.log1p(adata)

print("完成归一化。adata.X 现为 Log-Transformed 数据。")

# ================= 6. 特征选择与降维 =================
print("\nStep 6: 高变基因 (HVG) 与 PCA...")

# 寻找 Top 2000 高变基因
sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)

# 绘制高变基因图
sc.pl.highly_variable_genes(adata, show=False, save="_hvg.png")

# PCA 主成分分析
sc.pp.pca(adata, n_comps=50)

# 导出 PCA 坐标 (可选)
pd.DataFrame(adata.obsm['X_pca'], index=adata.obs_names,
             columns=[f'PC_{i + 1}' for i in range(50)]).to_csv(
    os.path.join(result_dir, "pca_coords.csv"))

# ================= 7. 聚类与 UMAP =================
print("\nStep 7: 聚类与 UMAP...")

# 计算邻近图
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=40)

# Leiden 聚类
sc.tl.leiden(adata, resolution=0.8, key_added="clusters")

# 计算 UMAP
sc.tl.umap(adata)

# 绘制 UMAP 组合图
plt.rcParams["figure.figsize"] = (4, 4)
sc.pl.umap(adata, color=["clusters", "total_counts", "n_genes_by_counts"],
           wspace=0.4, show=False, save="_combined.png")

# ================= 8. 空间可视化 (核心功能) =================
print("\nStep 8: 绘制空间分布图...")
plt.rcParams["figure.figsize"] = (8, 8)

# 8.1 空间聚类全景图
sc.pl.spatial(adata, img_key="hires", color="clusters", size=1.5,
              title="Spatial Clusters", show=False, save="_spatial_clusters.png")

# 8.2 局部裁剪图 (Crop)
# 注意：如果数据中没有 Cluster 1 或 3，此步会自动跳过。
try:
    print("尝试绘制局部裁剪图 (Cluster 1 & 3)...")
    sc.pl.spatial(
        adata,
        img_key="hires",
        color="clusters",
        groups=["1", "3"],  # 仅显示 Cluster 1 和 3
        crop_coord=[7000, 10000, 0, 6000],  # 裁剪坐标
        alpha=0.5,
        size=1.3,
        show=False,
        save="_spatial_crop.png"
    )
except Exception as e:
    print(f"局部裁剪图跳过: {e}")

# ================= 9. 差异分析与热图 =================
print("\nStep 9: 差异分析与热图...")

# 计算差异基因 (t-test)
sc.tl.rank_genes_groups(adata, 'clusters', method='t-test')

# 9.1 绘制热图 (Heatmap)
# 展示 Cluster 2 的 Top 10 差异基因
try:
    print("绘制 Cluster 2 热图...")
    sc.pl.rank_genes_groups_heatmap(
        adata, groups="2", n_genes=10, groupby="clusters",
        show=False, save="_heatmap_cluster2.png"
    )
except:
    pass

# 9.2 绘制矩阵点图 (Matrix Plot)
try:
    print("绘制 Matrix Plot...")
    markers = [x[0] for x in adata.uns['rank_genes_groups']['names'][:3]]
    sc.pl.matrixplot(adata, var_names=markers, groupby='clusters',
                     layer='cpm_counts', standard_scale='var',
                     show=False, save="_matrix_plot.png")
except:
    pass

# ================= 10. 数据导出 (工程化) =================
print("\nStep 10: 导出数据...")

# 10.1 保存完整存档 (H5AD)
h5ad_path = os.path.join(result_dir, "final_analysis.h5ad")
adata.write(h5ad_path)
print(f"完整分析存档已保存: {h5ad_path}")

# 10.2 导出元数据 CSV
adata.obs.to_csv(os.path.join(result_dir, "cell_metadata.csv"))

# 10.3 导出稀疏矩阵 (给 R 语言/同事用)
try:
    print("正在导出兼容 R 的稀疏矩阵...")
    mmwrite(os.path.join(matrix_dir, 'matrix.mtx'), adata.X)

    with open(os.path.join(matrix_dir, 'features.tsv'), 'w') as f:
        f.write("gene_id\tgene_name\n")
        for gene in adata.var_names:
            f.write(f"{gene}\t{gene}\n")

    with open(os.path.join(matrix_dir, 'barcodes.tsv'), 'w') as f:
        f.write("barcode\n")
        for bc in adata.obs_names:
            f.write(f"{bc}\n")
    print(f"矩阵已导出至: {matrix_dir}")
except Exception as e:
    print(f"矩阵导出跳过: {e}")


print("\n=== 所有分析流程结束 ===")
