# -*- coding: utf-8 -*-
"""
---------------------------------------------------------------------------
文件名称: st_analysis_advanced.py
功能描述: 空间转录组数据的分析流程（整合用户自定义画图与质控参数）
创建日期: 2026-02-01
作    者: Kuroneko

输入文件:
    1. 10x Genomics Visium 数据文件夹 (GBM1_spaceranger_out)

输出文件:
    1. results/GSE194329.data.anndata.h5ad (最终处理后的数据)
    2. results/figures/*.png (质控分布图、UMAP、空间分布图、热图等)
    3. results/hvg_genes.csv (高变基因表)

注意事项:
    - 代码中包含特定的 Cluster ID (如 '1', '2', '3') 筛选。
    - 如果数据重新聚类后 Cluster ID 发生变化，部分画图可能需要调整 ID。
---------------------------------------------------------------------------
"""

import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ================= 配置路径 =================
base_dir = r"F:\ST\code"
input_dir = os.path.join(base_dir, r"01\GSE194329_RAW\GBM1_spaceranger_out")
output_dir = os.path.join(base_dir, "results")
figure_dir = os.path.join(output_dir, "figures")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)

# 设置绘图保存路径
sc.settings.figdir = figure_dir
sc.set_figure_params(dpi=150, figsize=(6, 6))

print(f"Step 0: 路径配置完成。\n输入: {input_dir}\n输出: {output_dir}")

# ================= 1. 读取数据 =================
print("Step 1: 正在读取 Visium 数据...")
adata = sc.read_visium(input_dir, count_file='filtered_feature_bc_matrix.h5')
adata.var_names_make_unique()

# ================= 2. 质控指标计算与绘图 (QC Plots) =================
print("Step 2: 计算质控指标并绘制分布图...")

# 标记线粒体基因
adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

# 备份原始数据 (用户习惯)
adata1 = adata.copy()

# ---绘制质控分布直方图 ---
# 使用 seaborn 绘制 counts 和 gene numbers 的分布
fig, axs = plt.subplots(1, 4, figsize=(15, 4))

# 图1: 总 Counts 分布
sns.histplot(adata.obs["total_counts"], kde=False, ax=axs[0])
axs[0].set_title("Total Counts per Spot")

# 图2: Counts < 10000 的分布细节
sns.histplot(adata.obs["total_counts"][adata.obs["total_counts"] < 10000],
             kde=False, bins=40, ax=axs[1])
axs[1].set_title("Total Counts (<10k)")

# 图3: 基因数量分布
sns.histplot(adata.obs["n_genes_by_counts"], kde=False, bins=60, ax=axs[2])
axs[2].set_title("Genes per Spot")

# 图4: 基因数量 < 4000 的分布细节
sns.histplot(adata.obs["n_genes_by_counts"][adata.obs["n_genes_by_counts"] < 4000],
             kde=False, bins=60, ax=axs[3])
axs[3].set_title("Genes per Spot (<4k)")

plt.tight_layout()
# 保存质控分布图
plt.savefig(os.path.join(figure_dir, "qc_histograms.png"))
plt.close()
print("质控分布图已保存至 figures/qc_histograms.png")

# ================= 3. 数据过滤 (Filtering) =================
print("Step 3: 执行严格过滤 (User Defined Parameters)...")
print(f"过滤前细胞数: {adata.n_obs}")

# --- [用户自定义代码] 过滤参数 ---
sc.pp.filter_cells(adata, min_counts=5000)
sc.pp.filter_cells(adata, max_counts=35000)
adata = adata[adata.obs["pct_counts_mt"] < 20].copy()
sc.pp.filter_genes(adata, min_cells=10)

print(f"过滤后细胞数 (#cells after MT filter): {adata.n_obs}")

# ================= 4. 标准化与高变基因 =================
print("Step 4: 标准化、对数化与高变基因提取...")

sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)

# 提取 Top 2000 高变基因
sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)

# 保存高变基因表 (可选)
adata.var[adata.var['highly_variable']].to_csv(os.path.join(output_dir, "hvg_genes.csv"))

# ================= 5. 降维与聚类 =================
print("Step 5: PCA, Neighbors, UMAP, Leiden Clustering...")

sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
# 注意：key_added="clusters" 将聚类结果存为 adata.obs['clusters']
sc.tl.leiden(adata, key_added="clusters")

# ================= 6. UMAP 可视化 =================
print("Step 6: 绘制 UMAP 图...")

plt.rcParams["figure.figsize"] = (4, 4)

# 组合图
sc.pl.umap(adata, color=["total_counts", "n_genes_by_counts", "clusters"],
           wspace=0.4, show=False, save="_combined.png")

# 单独图
sc.pl.umap(adata, color=["total_counts"], wspace=0.4, show=False, save="_counts.png")
sc.pl.umap(adata, color=["n_genes_by_counts"], wspace=0.4, show=False, save="_ngenes.png")
sc.pl.umap(adata, color=["clusters"], wspace=0.4, show=False, save="_clusters.png")

# ================= 7. 空间可视化 (Spatial) =================
print("Step 7: 绘制空间分布图 (Spatial Plots)...")

plt.rcParams["figure.figsize"] = (8, 8)

# 基础空间图：Counts 和 Genes
sc.pl.spatial(adata, img_key="hires", color=["total_counts", "n_genes_by_counts"],
              show=False, save="_spatial_qc.png")

# 聚类空间图
sc.pl.spatial(adata, img_key="hires", color="clusters", size=1.5,
              show=False, save="_spatial_clusters.png")

# --- [用户自定义代码] 特定区域裁剪图 (Crop) ---
# 注意：这里筛选 groups=['1', '3']，如果聚类结果中没有这两个ID，可能会报错或显示空白
# 建议先检查 adata.obs['clusters'] 的类别
try:
    print("正在绘制特定裁剪区域 (Crop Coordinates)...")
    sc.pl.spatial(
        adata,
        img_key="hires",
        color="clusters",
        groups=["1", "3"],  # 只显示 Cluster 1 和 3
        crop_coord=[7000, 10000, 0, 6000],  # [x_min, x_max, y_min, y_max]
        alpha=0.5,
        size=1.3,
        show=False,
        save="_spatial_crop_1_3.png"
    )
except Exception as e:
    print(f"裁剪图绘制失败 (可能是Cluster ID 1或3不存在): {e}")

# ================= 8. 差异分析 (Diff Exp) =================
print("Step 8: 执行差异基因分析 (Rank Genes Groups)...")

# 使用 t-test 寻找差异基因
sc.tl.rank_genes_groups(adata, "clusters", method="t-test")

# --- [用户自定义代码] 绘制差异基因热图 ---
# 绘制 Cluster 2 的 Top 10 差异基因
try:
    print("正在绘制 Cluster 2 的差异基因热图...")
    sc.pl.rank_genes_groups_heatmap(
        adata,
        groups="2",  # 只展示 Cluster 2 的基因
        n_genes=10,
        groupby="clusters",
        show=False,
        save="_heatmap_cluster2.png"
    )
except Exception as e:
    print(f"热图绘制失败 (可能是Cluster 2不存在): {e}")

# ================= 9. 特定基因空间可视化 =================
print("Step 9: 绘制特定基因 (COL1A2, SYPL1) 的空间分布...")

# 再次绘制 clusters 作为对比
sc.pl.spatial(adata, img_key="hires", color=["clusters"], show=False, save="_spatial_clusters_ref.png")

# 绘制特定基因
genes_to_plot = ["COL1A2", "SYPL1"]
# 检查基因是否存在于数据中
valid_genes = [g for g in genes_to_plot if g in adata.var_names]

if valid_genes:
    sc.pl.spatial(
        adata,
        img_key="hires",
        color=valid_genes,
        alpha=0.7,
        show=False,
        save="_spatial_genes_COL1A2_SYPL1.png"
    )
else:
    print(f"警告: 基因 {genes_to_plot} 未在过滤后的数据中找到。")

# ================= 10. 保存结果 =================
print("Step 10: 保存最终数据...")

# 按照要求的文件名保存
save_path = os.path.join(output_dir, "GSE194329.data.anndata.h5ad")
adata.write(save_path)

print(f"\n全部完成！")
print(f"最终 h5ad 文件已保存至: {save_path}")
print(f"图片文件位于: {figure_dir}")