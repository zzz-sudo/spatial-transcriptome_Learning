# ===========================================================================
# File Name: 16_run_cell2location_analysis.py
# Author: Kuroneko
# Date: 2026-02-02
# Version: V2.5 (NMF Key Auto-Fix)
#
# Description:
#     [Cell2location 空间细胞类型映射全流程 - 终极完整版]
#
#     本脚本执行 Cell2location 的所有核心及高级分析功能。
#     已针对 Windows 系统优化，并修复了 NMF 分析中结果 Key 不匹配的报错。
#
#     核心流程 (Core Workflow):
#     1. [Data Loading]:
#        - 读取最原始的 Spatial 数据 (V1_Human_Lymph_Node) 和 SC 参考 (sc.h5ad)。
#        - 强调：必须使用 Raw Counts，不做归一化。
#     2. [Reference Model]:
#        - 训练模型学习每种细胞类型的基因表达特征 (Signature)。
#        - 自动处理 'Sample' 带来的批次效应。
#     3. [Spatial Mapping]:
#        - 训练模型推断每个 Spatial Spot 里的细胞绝对数量。
#     4. [Downstream - NMF]:
#        - (修复点) 自动识别 NMF 结果 Key，识别“空间微环境”。
#     5. [Downstream - Clustering]:
#        - 基于细胞丰度对 Spot 进行聚类，发现组织结构区域。
#     6. [Advanced Plotting]:
#        - 推断特定基因在特定细胞类型中的空间表达分布。
#     7. [Export]:
#        - 保存所有结果 (CSV表格, h5ad对象, 各种高清图片)。
#
# Input Files:
#     - Spatial: F:/ST/code/data/V1_Human_Lymph_Node/
#     - Reference: F:/ST/code/03/sc.h5ad
#
# Output Files (in F:/ST/code/results/cell2location_output/):
#     - 包含 reference_signatures, cell_abundance.csv, plots, h5ad 等所有文件。
# ===========================================================================

import sys
import os
import gc
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cell2location
from cell2location.utils.filtering import filter_genes
from cell2location.models import RegressionModel, Cell2location
import torch

# 设置绘图清晰度
sc.settings.set_figure_params(dpi=100, frameon=False, facecolor='white')
plt.rcParams['pdf.fonttype'] = 42

# -------------------------------------------------------------------------
# [Module 1] 环境初始化与路径设置
# -------------------------------------------------------------------------
print("[Module 1] Initializing Environment...")

# 使用 Windows 兼容路径
BASE_DIR = r"F:/ST/code/"
DATA_DIR = os.path.join(BASE_DIR, "data")
SC_FILE = os.path.join(BASE_DIR, "03", "sc.h5ad")
OUT_DIR = os.path.join(BASE_DIR, "results", "cell2location_output")

# 创建必要的子目录
REF_RUN_NAME = os.path.join(OUT_DIR, "reference_signatures")
SP_RUN_NAME = os.path.join(OUT_DIR, "spatial_mapping")
PLOT_DIR = os.path.join(OUT_DIR, "plots")

for d in [OUT_DIR, REF_RUN_NAME, SP_RUN_NAME, PLOT_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

print(f"    Output Directory: {OUT_DIR}")

# -------------------------------------------------------------------------
# [Module 2] 加载空间转录组数据 (必须是 Raw Data)
# -------------------------------------------------------------------------
print("\n[Module 2] Loading Spatial Data (Visium)...")

vis_path = os.path.join(DATA_DIR, "V1_Human_Lymph_Node")
if not os.path.exists(vis_path):
    raise FileNotFoundError(f"Spatial data not found at: {vis_path}")

# 读取 Visium 数据
adata_vis = sc.read_visium(path=vis_path, count_file='filtered_feature_bc_matrix.h5')
adata_vis.var_names_make_unique()

print(f"    Spatial Data: {adata_vis.shape[0]} spots x {adata_vis.shape[1]} genes")

# 预处理：移出线粒体基因 (减少干扰，这也是唯一的预处理)
adata_vis.var['MT_gene'] = [gene.startswith('MT-') for gene in adata_vis.var_names]
adata_vis.obsm['MT'] = adata_vis[:, adata_vis.var['MT_gene'].values].X.toarray()
adata_vis = adata_vis[:, ~adata_vis.var['MT_gene'].values]

# 添加 sample 列 (Cell2location 要求必须有 batch_key)
adata_vis.obs['sample'] = list(adata_vis.uns['spatial'].keys())[0]

# 绘制初始空间分布 QC 图
sc.pl.spatial(adata_vis, color=['in_tissue', 'array_col'], show=False)
plt.savefig(os.path.join(PLOT_DIR, "01_spatial_input_qc.png"), bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------------
# [Module 3] 加载与处理单细胞参考数据
# -------------------------------------------------------------------------
print("\n[Module 3] Loading & Filtering Single-Cell Reference...")

if not os.path.exists(SC_FILE):
    raise FileNotFoundError(f"Reference file not found: {SC_FILE}")

adata_ref = sc.read_h5ad(SC_FILE)
adata_ref.var_names_make_unique()

# 过滤基因：只保留在单细胞中具有区分度的基因
# 这步是 Cell2location 的核心特征选择步骤
print("    Filtering genes (Feature Selection)...")
selected = filter_genes(adata_ref, cell_count_cutoff=5, cell_percentage_cutoff2=0.03, nonz_mean_cutoff=1.12)
adata_ref = adata_ref[:, selected].copy()

print(f"    Reference Data Ready: {adata_ref.shape[0]} cells x {adata_ref.shape[1]} genes")

# -------------------------------------------------------------------------
# [Module 4] 训练参考模型 (Reference Model)
# -------------------------------------------------------------------------
print("\n[Module 4] Training Reference Model (Signature Extraction)...")

# 准备 AnnData
# 注意：这里模型会自动学习 'Sample' 列代表的批次效应，并将其移除，
# 从而提取出纯净的 'Subset' (细胞类型) 特征。
cell2location.models.RegressionModel.setup_anndata(
    adata=adata_ref,
    batch_key='Sample',
    labels_key='Subset',
    categorical_covariate_keys=['Method']
)

# 训练模型
# 移除了 use_gpu=True，自动检测 GPU
mod_ref = RegressionModel(adata_ref)
mod_ref.train(max_epochs=250, batch_size=2500, train_size=1, lr=0.002)

# 导出后验分布
adata_ref = mod_ref.export_posterior(
    adata_ref, sample_kwargs={'num_samples': 1000, 'batch_size': 2500}
)

# 提取并保存细胞签名表 (Signatures)
if 'means_per_cluster_mu_fg' in adata_ref.varm.keys():
    inf_aver = adata_ref.varm['means_per_cluster_mu_fg'][[f'means_per_cluster_mu_fg_{i}'
                                                          for i in adata_ref.uns['mod']['factor_names']]].copy()
else:
    inf_aver = adata_ref.var[[f'means_per_cluster_mu_fg_{i}'
                              for i in adata_ref.uns['mod']['factor_names']]].copy()
inf_aver.columns = adata_ref.uns['mod']['factor_names']
inf_aver.to_csv(os.path.join(OUT_DIR, "reference_signatures.csv"))

# 保存模型并清理显存
mod_ref.save(REF_RUN_NAME, overwrite=True)
del mod_ref, adata_ref
gc.collect()
torch.cuda.empty_cache()
print("    Reference Model Saved & GPU Memory Cleared.")

# -------------------------------------------------------------------------
# [Module 5] 空间映射 (Spatial Mapping)
# -------------------------------------------------------------------------
print("\n[Module 5] Mapping Cells to Space (Deconvolution)...")

# 取基因交集
intersect = np.intersect1d(adata_vis.var_names, inf_aver.index)
adata_vis = adata_vis[:, intersect].copy()
inf_aver = inf_aver.loc[intersect, :].copy()

# 准备空间数据
cell2location.models.Cell2location.setup_anndata(adata=adata_vis, batch_key="sample")

# 创建映射模型
# N_cells_per_location=30: 告诉模型这个组织的细胞密度比较大
mod_spatial = cell2location.models.Cell2location(
    adata_vis,
    cell_state_df=inf_aver,
    N_cells_per_location=30,
    detection_alpha=20
)

# 训练模型
print("    Training spatial model (this takes time)...")
mod_spatial.train(
    max_epochs=3000,
    batch_size=None,
    train_size=1
)

# 导出结果
print("    Exporting prediction results...")
adata_vis = mod_spatial.export_posterior(
    adata_vis, sample_kwargs={'num_samples': 1000, 'batch_size': mod_spatial.adata.n_obs}
)

# 保存模型
mod_spatial.save(SP_RUN_NAME, overwrite=True)

# 将预测结果 (5% 分位数) 写入 obs
adata_vis.obs[adata_vis.uns['mod']['factor_names']] = adata_vis.obsm['q05_cell_abundance_w_sf']

# 导出 CSV 表格 (最关键的数值结果)
abundance_df = pd.DataFrame(adata_vis.obsm['q05_cell_abundance_w_sf'], index=adata_vis.obs_names,
                            columns=adata_vis.uns['mod']['factor_names'])
abundance_df.to_csv(os.path.join(OUT_DIR, "cell_abundance.csv"))
print("    Spatial Mapping Complete. CSV Saved.")

# -------------------------------------------------------------------------
# [Module 6] 基础可视化
# -------------------------------------------------------------------------
print("\n[Module 6] Generating Basic Visualizations...")

# 1. QC 图
mod_spatial.plot_QC()
plt.savefig(os.path.join(PLOT_DIR, "02_reconstruction_QC.png"))
plt.close()

# 2. 空间分布图 (挑选几个代表性细胞)
target_cells = ['B_naive', 'B_GC_LZ', 'T_CD4+_naive', 'FDC', 'Endo']
plot_cells = [ct for ct in target_cells if ct in adata_vis.obs.columns]

if len(plot_cells) > 0:
    with mpl.rc_context({'axes.facecolor': 'black', 'figure.figsize': [4.5, 5]}):
        sc.pl.spatial(
            adata_vis,
            cmap='magma',
            color=plot_cells,
            ncols=3,
            size=1.3,
            img_key='hires',
            vmin=0, vmax='p99.2',
            show=False
        )
        plt.savefig(os.path.join(PLOT_DIR, "03_cell_type_maps.png"), bbox_inches='tight')
        plt.close()

# -------------------------------------------------------------------------
# [Module 7] 高级分析：NMF 识别共定位模式 (FIXED)
# -------------------------------------------------------------------------
print("\n[Module 7] Identifying Microenvironments (NMF Analysis)...")

from cell2location import run_colocation

# 运行 NMF，寻找 9 个共定位模块
# 注意：res_dict 会包含分析结果
res_dict, adata_vis = run_colocation(
    adata_vis,
    model_name='CoLocatedGroupsSklearnNMF',
    train_args={
        'n_fact': [9],
        'sample_name_col': 'sample',
        'n_restarts': 3
    },
    export_args={'path': os.path.join(OUT_DIR, 'co_located_groups')}
)

# [FIX] 自动寻找 NMF 结果 Key，防止 KeyError
print("    Searching for NMF output key in obsm...")
nmf_key = None
# 遍历所有 key，寻找包含 'nmf' 和 'fact' 的项
for key in adata_vis.obsm.keys():
    if 'nmf' in key and 'fact' in key:
        nmf_key = key
        print(f"    [Found] Key: {nmf_key}")
        break

if nmf_key:
    # 绘制结果
    if 'n_fact9' in res_dict:
        # 因子成分图 (Heatmap)
        res_dict['n_fact9']['mod'].plot_cell_type_loadings()
        plt.savefig(os.path.join(PLOT_DIR, "04_NMF_loadings.png"), bbox_inches='tight')
        plt.close()

    # 空间微环境图 (Spatial Map)
    n_factors = adata_vis.obsm[nmf_key].shape[1]
    factor_cols = [f'fact_{i}' for i in range(n_factors)]

    # 存入 obs 供绘图使用
    adata_vis.obs[factor_cols] = adata_vis.obsm[nmf_key]

    with mpl.rc_context({'axes.facecolor': 'black', 'figure.figsize': [4.5, 5]}):
        sc.pl.spatial(
            adata_vis,
            cmap='viridis',
            color=factor_cols,
            ncols=3,
            size=1.3,
            img_key='hires',
            show=False
        )
        plt.savefig(os.path.join(PLOT_DIR, "05_microenvironments_maps.png"), bbox_inches='tight')
        plt.close()
else:
    print("    [Warning] Could not find NMF result key in obsm. Skipping NMF plotting.")

# -------------------------------------------------------------------------
# [Module 8] 高级分析：下游聚类
# -------------------------------------------------------------------------
print("\n[Module 8] Clustering Spots based on Predicted Abundance...")

# 这里的逻辑是：既然我们已经知道了每个Spot里有多少种细胞，
# 那我们就可以根据“细胞组成”来给Spot分类（比如：生发中心区、边缘区等）
sc.pp.neighbors(adata_vis, use_rep='q05_cell_abundance_w_sf', n_neighbors=15)
sc.tl.leiden(adata_vis, resolution=0.5)
sc.tl.umap(adata_vis, min_dist=0.3, spread=1)

# UMAP
sc.pl.umap(adata_vis, color=['leiden', 'sample'], show=False)
plt.savefig(os.path.join(PLOT_DIR, "06_abundance_umap.png"), bbox_inches='tight')
plt.close()

# 空间聚类图
with mpl.rc_context({'axes.facecolor': 'black'}):
    sc.pl.spatial(
        adata_vis,
        color=['leiden'],
        size=1.3,
        img_key='hires',
        alpha=0.8,
        show=False
    )
    plt.savefig(os.path.join(PLOT_DIR, "07_spatial_clusters.png"), bbox_inches='tight')
    plt.close()

# -------------------------------------------------------------------------
# [Module 9] 高级绘图：特定基因的细胞特异性表达 (k13.md Extra)
# -------------------------------------------------------------------------
print("\n[Module 9] Computing Gene-Specific Spatial Expression...")

# 这个功能会计算：某个基因在某个位置的表达量，具体是由哪种细胞贡献的？
# 比如：CD3D 基因在位置 A 很高，到底是因为那里 T 细胞多，还是 B 细胞也表达了一点？

# 1. 计算期望表达量 (耗时步骤)
expected_dict = mod_spatial.module.model.compute_expected_per_cell_type(
    mod_spatial.samples["post_sample_q05"], mod_spatial.adata_manager
)

# 2. 将结果存入 layers
for i, n in enumerate(mod_spatial.factor_names_):
    adata_vis.layers[n] = expected_dict['mu'][i]

# 3. 绘图：针对 'CD3D' 基因，查看它在不同细胞类型中的贡献
# 这里我们模仿 k13.md 里的操作，画 CD3D 在 T细胞和 B细胞里的分布
gene_of_interest = 'CD3D'
cell_types_of_interest = ['T_CD4+_naive', 'B_GC_LZ']  # 对比这两种细胞

if gene_of_interest in adata_vis.var_names and all(c in adata_vis.obs.columns for c in cell_types_of_interest):
    with mpl.rc_context({'axes.facecolor': 'black', 'figure.figsize': [10, 4]}):
        fig, axs = plt.subplots(1, 3)

        # 图1: CD3D 的总体表达
        sc.pl.spatial(adata_vis, color=gene_of_interest, cmap='magma', img_key='hires', size=1.3, ax=axs[0], show=False,
                      title=f'Total {gene_of_interest}')

        # 图2: T细胞贡献的 CD3D
        sc.pl.spatial(adata_vis, color=gene_of_interest, layer='T_CD4+_naive', cmap='magma', img_key='hires', size=1.3,
                      ax=axs[1], show=False, title=f'{gene_of_interest} in T_Naive')

        # 图3: B细胞贡献的 CD3D (应该是黑的，因为B细胞不表达CD3D)
        sc.pl.spatial(adata_vis, color=gene_of_interest, layer='B_GC_LZ', cmap='magma', img_key='hires', size=1.3,
                      ax=axs[2], show=False, title=f'{gene_of_interest} in B_GC')

        plt.savefig(os.path.join(PLOT_DIR, "08_gene_specific_expression_CD3D.png"), bbox_inches='tight')
        plt.close()
    print("    Gene specific plot saved.")
else:
    print(f"    [Skip] {gene_of_interest} or cell types not found.")

# -------------------------------------------------------------------------
# [Module 10] 保存最终结果
# -------------------------------------------------------------------------
print("\n[Module 10] Saving Final AnnData Object...")

final_h5ad_path = os.path.join(OUT_DIR, "sp_with_results.h5ad")
adata_vis.write(final_h5ad_path)

print("=" * 60)
print(f"Cell2location Analysis Completed Successfully.")
print(f"Results are saved in: {OUT_DIR}")
print("=" * 60)