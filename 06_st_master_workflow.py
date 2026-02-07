# -*- coding: utf-8 -*-
"""
===========================================================================
文件名称: st_grand_master_final.py
作    者: Kuroneko

版    本: V18.0 (Final Release)

功能描述:
    [空间转录组全流程系统性分析平台]
    本脚本旨在提供一套完整的、端到端的空间转录组分析解决方案。
    核心功能涵盖从原始数据读取到高级可视化的全链路处理:

    1. 数据预处理与质控 (Data Preprocessing & QC):
       - 加载多样本切片数据，计算线粒体指标。
       - 执行严格的质控过滤，剔除低质量细胞与背景噪点。

    2. 多模态整合 (Integration):
       - 对不同切片进行标准化拼接，恢复空间坐标信息。
       - 计算高变基因 (HVG) 并应用 PCA 降维提取核心特征。

    3. 批次效应评估与矫正 (Batch Correction):
       - [新增] 绘制矫正前的 UMAP 分布，直观评估样本间的技术差异。
       - 应用 Harmony 算法进行批次矫正，消除切片间的系统性误差。
       - 绘制矫正后的 UMAP 分布，验证多样本整合效果。

    4. 细胞聚类与注释 (Clustering & Annotation):
       - 基于矫正后的特征进行 Leiden 高分辨率聚类。
       - 集成单细胞参考集(scRNA-seq)，利用 Ingest 算法实现高精度的空间细胞类型注释。

    5. 差异表达分析 (DGE Analysis):
       - 识别各个细胞簇及亚群的特异性 Marker 基因。

    6. 综合可视化图谱 (Advanced Visualization):
       - 生成包含 QC、批次对比、全景空间分布、局部微环境裁剪、
       - 热图 (Heatmap)、气泡图 (Dotplot)、堆叠小提琴图 (Stacked Violin) 等 12 种专业图表。

输入文件 (Input):
    1. F:/ST/code/02/adata_spatial_anterior.h5ad
    2. F:/ST/code/02/adata_spatial_posterior.h5ad
    3. F:/ST/code/02/adata_processed.h5ad

输出文件 (Output):
    1. F:/ST/code/results/grand_master_final.h5ad
    2. F:/ST/code/results/figures_grand_master/
===========================================================================
"""

import os
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import harmonypy as hm

# =========================================================================
# [Module 0] 系统环境初始化
# =========================================================================
print("[Module 0] 初始化系统环境...")

BASE_DIR = r"F:\ST\code"
INPUT_DIR = os.path.join(BASE_DIR, "02")
RESULT_DIR = os.path.join(BASE_DIR, "results")
FIGURE_DIR = os.path.join(RESULT_DIR, "figures_grand_master")

if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)
if not os.path.exists(FIGURE_DIR):
    os.makedirs(FIGURE_DIR)

sc.settings.verbosity = 3
sc.settings.figdir = FIGURE_DIR
sc.set_figure_params(dpi=150, facecolor="white", vector_friendly=True)
plt.rcParams['font.family'] = 'sans-serif'

print(f"当前工作目录: {BASE_DIR}")
print(f"结果输出目录: {RESULT_DIR}")
print(f"图表保存目录: {FIGURE_DIR}")

# =========================================================================
# [Module A] 数据加载与基础质控
# =========================================================================
print("\n[Module A] 开始加载数据并执行质控...")


def load_and_qc_sample(filename, batch_label, sample_title):
    file_path = os.path.join(INPUT_DIR, filename)
    print(f"[INFO] 读取文件: {filename} ({sample_title})")

    try:
        adata = sc.read_h5ad(file_path)
    except Exception as e:
        print(f"[ERROR] 读取失败: {e}")
        return None

    adata.var_names_make_unique()
    adata.obs['batch'] = batch_label

    # 计算 QC
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

    # 绘图: 质控图
    sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
                 jitter=0.4, multi_panel=True, show=False,
                 save=f"_QC_Violin_{batch_label}.png")

    sc.pl.spatial(adata, img_key="hires", color=["total_counts", "n_genes_by_counts"],
                  size=1.5, show=False,
                  save=f"_QC_Spatial_{batch_label}.png")

    # 过滤
    print(f"    [Filter] 过滤前斑点数: {adata.n_obs}")
    sc.pp.filter_cells(adata, min_counts=500)
    adata = adata[adata.obs["pct_counts_mt"] < 25].copy()
    print(f"    [Filter] 过滤后斑点数: {adata.n_obs}")

    # 归一化
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


adata_ant = load_and_qc_sample("adata_spatial_anterior.h5ad", "Batch_Ant", "Anterior")
adata_post = load_and_qc_sample("adata_spatial_posterior.h5ad", "Batch_Post", "Posterior")

if adata_ant is None or adata_post is None:
    print("[ERROR] 关键数据缺失，程序终止。")
    exit()

# =========================================================================
# [Module B] 多样本整合与 Harmony 去批次
# =========================================================================
print("\n[Module B] 执行多样本整合与 Harmony 去批次...")

# 1. 拼接
print("[Step 1] 拼接样本数据...")
adata_spatial = sc.concat([adata_ant, adata_post], label="batch",
                          keys=["Batch_Ant", "Batch_Post"], join="outer")

# 2. 恢复图像
adata_spatial.uns['spatial'] = {}
if 'spatial' in adata_ant.uns:
    adata_spatial.uns['spatial'].update(adata_ant.uns['spatial'])
if 'spatial' in adata_post.uns:
    adata_spatial.uns['spatial'].update(adata_post.uns['spatial'])

# 3. PCA
print("[Step 2] 计算高变基因与 PCA...")
sc.pp.highly_variable_genes(adata_spatial, flavor="seurat", n_top_genes=2000, batch_key="batch")
sc.pp.pca(adata_spatial, n_comps=50)

# -------------------------------------------------------------------------
# [新增] 绘制 "去批次前" (Before Harmony) 的 UMAP
# -------------------------------------------------------------------------
print("[Step 2.5] 评估去批次前的分布 (Before Harmony)...")
# 使用原始 PCA 计算邻居
sc.pp.neighbors(adata_spatial, use_rep='X_pca', n_neighbors=15)
sc.tl.umap(adata_spatial)
# 绘图: 展示原始批次效应
sc.pl.umap(adata_spatial, color="batch", title="Batch Effect (Before Harmony)",
           show=False, save="_Batch_Before_Harmony.png")
print("    [Info] 已生成去批次前的 UMAP 对比图。")

# 4. Harmony
print("[Step 3] 运行 Harmony 算法 (底层接口模式)...")
pca_matrix = adata_spatial.obsm['X_pca']
batch_meta = adata_spatial.obs

# 执行计算
ho = hm.run_harmony(pca_matrix, batch_meta, 'batch', max_iter_harmony=20)

# 智能形状检测
harmony_out = ho.Z_corr
target_rows = adata_spatial.n_obs

if harmony_out.shape[0] == target_rows:
    print("    [Info] 矩阵形状正确，直接应用。")
    final_harmony = harmony_out
else:
    print("    [Info] 矩阵形状需要转置，正在修正...")
    final_harmony = harmony_out.T

adata_spatial.obsm['X_pca_harmony'] = np.array(final_harmony)
use_rep_key = 'X_pca_harmony'

# 5. 聚类 (基于去批次后)
print("[Step 4] 基于 Harmony 结果执行 UMAP 和 Leiden 聚类...")
sc.pp.neighbors(adata_spatial, use_rep=use_rep_key, n_neighbors=15)
sc.tl.umap(adata_spatial)
sc.tl.leiden(adata_spatial, resolution=0.6, key_added="leiden_harmony")

# 绘图: 展示去批次后的效果
sc.pl.umap(adata_spatial, color=["batch", "leiden_harmony"], wspace=0.4,
           title=["Batch Effect (After Harmony)", "Harmony Clusters"],
           show=False, save="_Batch_After_Harmony.png")

# =========================================================================
# [Module C] 单细胞辅助注释
# =========================================================================
print("\n[Module C] 执行单细胞辅助注释...")

# 1. 读取参考集
ref_path = os.path.join(INPUT_DIR, "adata_processed.h5ad")
print(f"[Info] 读取参考集: {ref_path}")
adata_ref = sc.read_h5ad(ref_path)
adata_ref.var_names_make_unique()

ref_label = 'cell_subclass'
if ref_label not in adata_ref.obs.columns:
    ref_label = adata_ref.obs.columns[0]
print(f"    使用参考集标签: {ref_label}")

# 2. 基因对齐
common_genes = adata_ref.var_names.intersection(adata_spatial.var_names)
print(f"    共有基因数量: {len(common_genes)}")
adata_ref = adata_ref[:, common_genes].copy()
adata_spatial = adata_spatial[:, common_genes].copy()

# 3. 训练参考集
print("[Info] 重新训练参考集 (PCA -> Neighbors -> UMAP)...")
if 'log1p' not in adata_ref.uns:
    sc.pp.normalize_total(adata_ref, target_sum=1e4)
    sc.pp.log1p(adata_ref)

sc.pp.pca(adata_ref)
sc.pp.neighbors(adata_ref)
sc.tl.umap(adata_ref)

# 4. Ingest
print("[Info] 执行 Ingest 投影...")
sc.tl.ingest(adata_spatial, adata_ref, obs=ref_label)

# 绘图
sc.pl.umap(adata_spatial, color=ref_label, title="Projected Cell Types",
           show=False, save="_Projected_UMAP.png")

# =========================================================================
# [Module D] 差异基因分析
# =========================================================================
print("\n[Module D] 计算差异 Marker 基因...")

marker_group = "leiden_harmony"
sc.tl.rank_genes_groups(adata_spatial, groupby=marker_group, method='t-test')

markers = [x[0] for x in adata_spatial.uns['rank_genes_groups']['names'][:3]]
markers = list(dict.fromkeys(markers))

print("Top 5 Marker 基因示例:")
print(pd.DataFrame(adata_spatial.uns['rank_genes_groups']['names']).head(5))

# =========================================================================
# [Module E] 高级可视化图表生成
# =========================================================================
print("\n[Module E] 生成全套高级可视化图表...")

# [E1] 全景对比图
print("--> [E1] 绘制全景对比图...")
fig, axs = plt.subplots(1, 2, figsize=(16, 7))

sub_ant = adata_spatial[adata_spatial.obs['batch'] == 'Batch_Ant']
lib_ant = list(sub_ant.uns['spatial'].keys())[0]
sc.pl.spatial(sub_ant, library_id=lib_ant, img_key="hires", color=ref_label,
              size=1.4, title="Anterior", ax=axs[0], show=False, legend_loc=None)

sub_post = adata_spatial[adata_spatial.obs['batch'] == 'Batch_Post']
lib_post = list(sub_post.uns['spatial'].keys())[0]
sc.pl.spatial(sub_post, library_id=lib_post, img_key="hires", color=ref_label,
              size=1.4, title="Posterior", ax=axs[1], show=False)

plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "E1_Spatial_Prediction_Full.png"))
plt.close()

# [E2] 局部裁剪图
print("--> [E2] 绘制局部裁剪图...")
crop_coords = [3000, 7000, 3000, 7000]
sc.pl.spatial(sub_ant, library_id=lib_ant, img_key="hires",
              color=[ref_label, marker_group],
              crop_coord=crop_coords,
              size=2.0, title=["Cropped CellType", "Cropped Cluster"],
              show=False, save="_E2_Spatial_Cropped.png")

# [E3] 热图
print("--> [E3] 绘制差异基因热图...")
sc.pl.rank_genes_groups_heatmap(adata_spatial, n_genes=5, groupby=marker_group,
                                show_gene_labels=True, swap_axes=True, figsize=(10, 15),
                                dendrogram=False, show=False, save="_E3_Heatmap.png")

# [E4] 气泡图
print("--> [E4] 绘制基因气泡图...")
sc.pl.dotplot(adata_spatial, var_names=markers, groupby=marker_group,
              standard_scale='var', show=False, save="_E4_Dotplot.png")

# [E5] 堆叠小提琴图
print("--> [E5] 绘制堆叠小提琴图...")
sc.pl.stacked_violin(adata_spatial, var_names=markers, groupby=marker_group,
                     swap_axes=True, show=False, save="_E5_Stacked_Violin.png")

# [E6] 单基因空间分布图
print("--> [E6] 绘制单基因空间分布图...")
gene_demo = markers[0]
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
sc.pl.spatial(sub_ant, library_id=lib_ant, img_key="hires", color=gene_demo,
              size=1.5, cmap='Reds', title=f"{gene_demo} (Ant)",
              ax=axs[0], show=False)
sc.pl.spatial(sub_ant, library_id=lib_ant, img_key="hires", color=markers[1],
              size=1.5, cmap='Blues', title=f"{markers[1]} (Ant)",
              ax=axs[1], show=False)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "E6_Gene_Spatial_Expression.png"))
plt.close()

# [E7] Top 4 细胞类型分面图
print("--> [E7] 绘制 Top 4 细胞类型分面图...")
top_types_list = adata_spatial.obs[ref_label].value_counts().index[:4].tolist()
print(f"    Top 4 类型: {top_types_list}")

print("    绘制 Anterior 分面图...")
sc.pl.spatial(sub_ant, library_id=lib_ant, img_key="hires",
              color=ref_label, groups=top_types_list,
              size=1.5, title="Top 4 Types (Anterior)",
              show=False, save="_E7_Top4_Split_Anterior.png")

print("    绘制 Posterior 分面图...")
sc.pl.spatial(sub_post, library_id=lib_post, img_key="hires",
              color=ref_label, groups=top_types_list,
              size=1.5, title="Top 4 Types (Posterior)",
              show=False, save="_E7_Top4_Split_Posterior.png")

# =========================================================================
# [Module F] 结果保存
# =========================================================================
print("\n[Module F] 正在保存分析结果...")
save_path = os.path.join(RESULT_DIR, "grand_master_final.h5ad")
adata_spatial.write(save_path, compression="gzip")

print("-" * 60)
print("恭喜！[空间转录组大师级流程] 已成功执行完毕。")
print(f"结果数据: {save_path}")
print(f"全套图谱: {FIGURE_DIR}")

print("-" * 60)
