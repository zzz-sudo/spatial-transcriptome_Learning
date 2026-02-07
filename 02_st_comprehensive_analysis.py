# -*- coding: utf-8 -*-
"""
---------------------------------------------------------------------------
文件名称: st_comprehensive_analysis.py
作    者: Kuroneko
创建日期: 2026-02-01
功能描述:
    1. 空间转录组全流程分析 (读取, QC, 过滤, 归一化, 降维, 聚类, 空间绘图).
    2. 深度解析 AnnData 数据结构 (X, layers, obs, obsm 的操作演示).
    3. 导出多种格式结果 (csv, mtx, h5ad).

输入文件:
    F:/ST/code/01/GSE194329_RAW/GBM1_spaceranger_out (10x Visium 文件夹)

输出文件:
    F:/ST/code/results/ (包含图片, csv表格, 稀疏矩阵, h5ad文件)
---------------------------------------------------------------------------
"""
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from scipy import sparse
from scipy.io import mmwrite

# ================= 1. 环境与路径设置 =================
print("Step 1: 环境配置与路径检查...")

base_dir = r"F:\ST\code"
input_dir = os.path.join(base_dir, r"01\GSE194329_RAW\GBM1_spaceranger_out")
result_dir = os.path.join(base_dir, "results")
figure_dir = os.path.join(result_dir, "figures")
matrix_dir = os.path.join(result_dir, "sparse_matrix")

for path in [result_dir, figure_dir, matrix_dir]:
    if not os.path.exists(path):
        os.makedirs(path)

# Scanpy 绘图设置
sc.settings.verbosity = 3
sc.settings.figdir = figure_dir
sc.set_figure_params(dpi=150, facecolor="white", figsize=(8, 8))

print(f"输入目录: {input_dir}")
print(f"输出目录: {result_dir}")

# ================= 2. 数据读取与基础探索 =================
print("\nStep 2: 读取数据并探索 AnnData 结构...")

try:
    adata = sc.read_visium(input_dir, count_file='filtered_feature_bc_matrix.h5')
    adata.var_names_make_unique()
except Exception as e:
    print(f"读取失败: {e}")
    exit()

# 备份原始计数 (非常重要)
adata.layers["raw_counts"] = adata.X.copy()

# ---------------------------------------------------------
# [学习模块 - 可选删除] 探索 .X 底层结构与内存占用
# ---------------------------------------------------------
print("\n--- [学习模块 Start] 探索表达矩阵 .X ---")
print(f"X 的数据类型: {type(adata.X)}")

# 计算内存占用对比 (展示稀疏矩阵的优势)
try:
    # 注意：只取一小部分计算，避免撑爆内存
    small_sample = adata.X[:1000, :1000]
    dense_size = sys.getsizeof(small_sample.toarray()) / (1024 * 1024)
    sparse_size = sys.getsizeof(small_sample) / (1024 * 1024)
    print(f"1000x1000 矩阵 - 稠密格式占用: {dense_size:.2f} MB")
    print(f"1000x1000 矩阵 - 稀疏格式占用: {sparse_size:.2f} MB")
except:
    pass

print("X 矩阵左上角 3x3 数据示例:")
print(adata.X[:3, :3].toarray())
print("--- [学习模块 End] ---\n")
# ---------------------------------------------------------

# ================= 3. 高级质控 (QC) =================
print("\nStep 3: 计算质控指标并绘制高级图...")

adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

# [高级绘图] Seaborn 质控直方图
fig, axs = plt.subplots(1, 4, figsize=(16, 4))
sns.histplot(adata.obs["total_counts"], kde=False, ax=axs[0])
axs[0].set_title("Total Counts")
sns.histplot(adata.obs["total_counts"][adata.obs["total_counts"] < 10000], kde=False, bins=40, ax=axs[1])
axs[1].set_title("Counts < 10k")
sns.histplot(adata.obs["n_genes_by_counts"], kde=False, bins=60, ax=axs[2])
axs[2].set_title("Genes per Spot")
sns.histplot(adata.obs["pct_counts_mt"], kde=False, bins=60, ax=axs[3])
axs[3].set_title("Mitochondrial %")
plt.tight_layout()
plt.savefig(os.path.join(figure_dir, "01_qc_histograms_seaborn.png"))
plt.close()

# 空间 QC 图
sc.pl.spatial(adata, img_key="hires", color=["total_counts", "pct_counts_mt"],
              show=False, save="_qc_spatial.png")

# ================= 4. 严格数据过滤 =================
print("\nStep 4: 执行严格数据过滤...")
print(f"过滤前: {adata.n_obs} spots")

# [用户指定参数]
sc.pp.filter_cells(adata, min_counts=5000)
sc.pp.filter_cells(adata, max_counts=35000)
adata = adata[adata.obs["pct_counts_mt"] < 20, :].copy()
sc.pp.filter_genes(adata, min_cells=10)

print(f"过滤后: {adata.n_obs} spots")

# ================= 5. 归一化与 Layers =================
print("\nStep 5: 归一化与对数化...")

sc.pp.normalize_total(adata, target_sum=1e4)
adata.layers["cpm_counts"] = adata.X.copy()  # 备份标准化后的数据
sc.pp.log1p(adata)

# ---------------------------------------------------------
# [学习模块 - 可选删除] 探索 Layers
# ---------------------------------------------------------
print("\n--- [学习模块 Start] 探索 Layers ---")
print(f"当前存在的图层: {adata.layers.keys()}")
print("原始值:", adata.layers['raw_counts'][0, 0])
print("归一化后:", adata.layers['cpm_counts'][0, 0])
print("对数化后(.X):", adata.X[0, 0])
print("--- [学习模块 End] ---\n")
# ---------------------------------------------------------

# ================= 6. 特征选择与降维 =================
print("\nStep 6: 高变基因 (HVG) 与 PCA...")

sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)
sc.pl.highly_variable_genes(adata, show=False, save="_hvg.png")

sc.pp.pca(adata, n_comps=50)

# [教学点] 导出 PCA 坐标
pd.DataFrame(adata.obsm['X_pca'], index=adata.obs_names).to_csv(
    os.path.join(result_dir, "pca_coords.csv"))

# ================= 7. 聚类与 UMAP =================
print("\nStep 7: 聚类与 UMAP...")

sc.pp.neighbors(adata, n_neighbors=15, n_pcs=40)
sc.tl.leiden(adata, resolution=0.8, key_added="clusters")
sc.tl.umap(adata)

# ---------------------------------------------------------
# [学习模块 - 可选删除] 手动 Matplotlib 绘制 UMAP
# ---------------------------------------------------------
print("\n--- [学习模块 Start] 手动复现 UMAP ---")
print("正在使用 matplotlib 手动绘制 UMAP (不使用 scanpy 函数)...")
plt.figure(figsize=(6, 6))
umap_coords = adata.obsm['X_umap']
# 简单绘制散点，验证 .obsm 数据的真实性
plt.scatter(umap_coords[:, 0], umap_coords[:, 1], s=3, c='gray', alpha=0.5)
plt.title("Manual UMAP (Raw Coordinates from .obsm)")
plt.xlabel("UMAP_1")
plt.ylabel("UMAP_2")
plt.savefig(os.path.join(figure_dir, "debug_manual_umap.png"))
plt.close()
print("手动 UMAP 已保存至 figures/debug_manual_umap.png")
print("--- [学习模块 End] ---\n")
# ---------------------------------------------------------

# Scanpy 标准 UMAP
sc.pl.umap(adata, color=["clusters", "total_counts"], wspace=0.4,
           show=False, save="_combined.png")

# ================= 8. 空间可视化 (全功能) =================
print("\nStep 8: 绘制空间分布图 (含局部裁剪)...")
plt.rcParams["figure.figsize"] = (8, 8)

# 8.1 标准全景图
sc.pl.spatial(adata, img_key="hires", color="clusters", size=1.5,
              title="Spatial Clusters", show=False, save="_spatial_clusters.png")

# 8.2 [高级绘图] 局部裁剪 (Crop)
# 注意：如果找不到 Cluster 1 或 3，会跳过
try:
    print("尝试绘制局部裁剪图 (Cluster 1 & 3)...")
    sc.pl.spatial(
        adata,
        img_key="hires",
        color="clusters",
        groups=["1", "3"],  # 只显示 Cluster 1 和 3
        crop_coord=[7000, 10000, 0, 6000],  # [xmin, xmax, ymin, ymax]
        alpha=0.5,
        size=1.3,
        show=False,
        save="_spatial_crop.png"
    )
except Exception as e:
    print(f"局部裁剪图绘制跳过: {e}")

# ================= 9. 差异分析与热图 =================
print("\nStep 9: 差异分析与热图...")

sc.tl.rank_genes_groups(adata, 'clusters', method='t-test')

# [高级绘图] 绘制热图 (Heatmap)
# 展示 Cluster 2 的 Top 10 差异基因
try:
    print("绘制 Cluster 2 热图...")
    sc.pl.rank_genes_groups_heatmap(
        adata, groups="2", n_genes=10, groupby="clusters",
        show=False, save="_heatmap_cluster2.png"
    )
except:
    pass

# [高级绘图] 绘制 Matrix Plot (使用标准化 Counts)
try:
    print("绘制 Matrix Plot...")
    # 获取前几个 marker 基因
    markers = [x[0] for x in adata.uns['rank_genes_groups']['names'][:3]]
    sc.pl.matrixplot(adata, var_names=markers, groupby='clusters',
                     layer='cpm_counts', standard_scale='var',
                     show=False, save="_matrix_plot.png")
except:
    pass

# ================= 10. 数据导出 (工程化) =================
print("\nStep 10: 导出数据 (工程化)...")

# 10.1 保存完整存档 (Python 用)
h5ad_path = os.path.join(result_dir, "GSE194329.data.anndata.h5ad")
adata.write(h5ad_path)
print(f"完整分析存档: {h5ad_path}")

# 10.2 导出 CSV (Excel 用)
adata.obs.to_csv(os.path.join(result_dir, "cell_metadata.csv"))

# ---------------------------------------------------------
# [可选模块 - 可选删除] 导出稀疏矩阵 (R语言/同事用)
# ---------------------------------------------------------
try:
    print("正在导出兼容 R 的稀疏矩阵 (matrix.mtx)...")
    mmwrite(os.path.join(matrix_dir, 'matrix.mtx'), adata.X)

    # 导出基因名 (使用高效写法)
    with open(os.path.join(matrix_dir, 'features.tsv'), 'w') as f:
        f.write("gene_id\tgene_name\n")
        for gene in adata.var_names:
            f.write(f"{gene}\t{gene}\n")

    # 导出 Barcodes
    with open(os.path.join(matrix_dir, 'barcodes.tsv'), 'w') as f:
        f.write("barcode\n")
        for bc in adata.obs_names:
            f.write(f"{bc}\n")

    print("导出完成: results/sparse_matrix/")
except Exception as e:
    print(f"矩阵导出跳过: {e}")
# ---------------------------------------------------------

print("\n=== 所有分析结束 ===")