# ===========================================================================
# File Name: 21_run_commot_analysis.py
# Author: Kuroneko
# Date: 2026-02-03
# Version: V1.0 (Optimal Transport Edition)
#
# Description:
#     [COMMOT 空间通讯分析 - 深度版]
#
#     核心功能:
#     1. 空间通讯推断: 使用最优传输 (OT) 理论计算配体-受体信号对。
#     2. 空间约束: 设定 dis_thr (物理距离阈值) 来模拟信号扩散极限。
#     3. 方向性分析: 推断信号是从哪里流向哪里的 (Signaling Direction)。
#     4. 下游靶点: (可选) 分析信号通路对下游基因的影响。
#
# Input Files:
#     - Data Dir: F:/ST/code/data/V1_Human_Lymph_Node/
#
# Output Files (in F:/ST/code/results/commot/):
#     - commot.h5ad: 包含通讯结果的 AnnData 对象
#     - plots/: 信号流向图、通讯强度热图
'''
stLearn: 综合形态学特征的分析。
SPATA2: 轨迹和去噪分析。
CellChat: 经典的配受体通讯。
COMMOT: 具有物理流向的空间通讯。
MISTy: 现在的这个，用于量化细胞类型间的空间依赖性。'''
# ===========================================================================

import os
import gc
import sys
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import commot as ct
import warnings

# 忽略部分不必要的警告
warnings.filterwarnings("ignore")

# -------------------------------------------------------------------------
# [Module 1] 环境与路径初始化
# -------------------------------------------------------------------------
print("[Module 1] Initializing Environment...")

BASE_DIR = r"F:/ST/code/"
DATA_DIR = os.path.join(BASE_DIR, "data", "V1_Human_Lymph_Node")
OUT_DIR = os.path.join(BASE_DIR, "results", "commot")
PLOT_DIR = os.path.join(OUT_DIR, "plots")

for d in [OUT_DIR, PLOT_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

# 设置绘图风格
sc.settings.figdir = PLOT_DIR
sc.set_figure_params(dpi=120, facecolor="white")

# -------------------------------------------------------------------------
# [Module 2] 数据加载与预处理
# -------------------------------------------------------------------------
print("[Module 2] Loading Visium Data...")

# 读取 10x Visium 数据
# Scanpy 会自动读取 spatial 文件夹下的图像和缩放因子
adata = sc.read_visium(DATA_DIR)
adata.var_names_make_unique()

# 标准化预处理 (COMMOT 需要 Normalized 数据，不要 Scale)
sc.pp.calculate_qc_metrics(adata, inplace=True)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# 备份原始数据用于可视化
adata.raw = adata

print(f"    Data Loaded: {adata.shape[0]} spots x {adata.shape[1]} genes")

# -------------------------------------------------------------------------
# [Module 3] 准备配体-受体数据库 (双重重命名版)
# -------------------------------------------------------------------------
print("[Module 3] Preparing Ligand-Receptor Database...")

# 1. 加载数据库
df_cellchat = ct.pp.ligand_receptor_database(species='human', signaling_type='Secreted Signaling')

# 定义标准列名
target_cols = ['ligand', 'receptor', 'pathway_name', 'annotation']

# [FIX 1] 过滤前重命名 (虽然可能被过滤函数重置，但为了保险先改了)
if len(df_cellchat.columns) == 4:
    df_cellchat.columns = target_cols

# 2. 过滤数据库
print("    Filtering database...")
df_lr_filtered = ct.pp.filter_lr_database(df_cellchat, adata, min_cell_pct=0.05)

# [FIX 2] 过滤后再次强制重命名 (这步才是关键！解决过滤后表头丢失的问题)
print("    [Fixing] Re-applying column names after filtering...")
if len(df_lr_filtered.columns) == 4:
    df_lr_filtered.columns = target_cols
    print(f"    [Check] Filtered DF Columns: {df_lr_filtered.columns.tolist()}")
else:
    print(f"    [Warning] Filtered DF has unexpected column count: {len(df_lr_filtered.columns)}")

print(f"    Original Pairs: {df_cellchat.shape[0]}")
print(f"    Filtered Pairs (Expressed in Data): {df_lr_filtered.shape[0]}")


# -------------------------------------------------------------------------
# [Module 4] 运行空间通讯推断 (Optimal Transport)
# -------------------------------------------------------------------------
print("[Module 4] Running COMMOT Spatial Communication (This may take time)...")

# 核心参数解释:
# dis_thr: 距离阈值 (单位: 像素或微米，取决于坐标系)。
#          Visium spot 间距约 100um。设置 500 表示允许信号传播约 5 个 spot 的距离。
#          注意: Scanpy 读取的 spatial 坐标通常是像素坐标，需要根据 scalefactors 换算。
#          这里我们设定一个经验值，大约覆盖周围 3-5 圈 spot。

# 这里的 500 是基于像素坐标的经验值，具体可视 scalefactors 而定
# 如果数据很大，建议先只跑前 50 个通路进行测试
df_lr_run = df_lr_filtered.head(50)
print(f"    Running on top {len(df_lr_run)} LR pairs for demonstration...")

# 运行主要函数
ct.tl.spatial_communication(
    adata,
    database_name='cellchat',
    df_ligrec=df_lr_run,
    dis_thr=500,    # 空间距离限制
    heteromeric=True, # 考虑复合物
    pathway_sum=True  # 计算通路总和
)

print("    Communication inference done.")

# -------------------------------------------------------------------------
# [Module 5] 批量信号方向性推断与绘图 (箭头放大版)
# -------------------------------------------------------------------------
print("[Module 5] Inferring Signaling Direction (Bigger Arrows)...")

pathway_col = 'pathway_name'
all_pathways = df_lr_run[pathway_col].unique()

# [设置] 依然先只画前 10 个看看效果
target_pathways = all_pathways[:10]

# [关键调整] 箭头比例尺
# 原来是 0.00003 (太小)
# 尝试增大到 0.0002 或 0.0005。数值越大，箭头越大。
ARROW_SCALE = 0.0005

# [可选调整] 背景图透明度 (0.0 - 1.0)
# 如果背景太深喧宾夺主，可以把这个调小，比如 0.5
BACKGROUND_ALPHA = 0.8

print(f"    -> Will process top {len(target_pathways)} pathways with scale={ARROW_SCALE}")

if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

for i, pathway in enumerate(target_pathways):
    print(f"    [{i + 1}/{len(target_pathways)}] Processing: {pathway} ...")

    try:
        # A. 计算 (如果之前算过了，这一步其实很快)
        ct.tl.communication_direction(adata, database_name='cellchat', pathway_name=pathway, k=5)

        # B. 绘图
        plt.figure(figsize=(10, 10))

        ct.pl.plot_cell_communication(
            adata,
            database_name='cellchat',
            pathway_name=pathway,
            plot_method='grid',
            background_legend=True,

            # [修改点 1] 使用更大的箭头比例
            scale=ARROW_SCALE,

            ndsize=8,
            grid_density=0.4,
            summary='sender',
            background='image',

            # [修改点 2] (可选) 设置背景图透明度，需要 commot 新版支持
            # 如果报错说没有 alpha_img 参数，请删掉这一行
            # alpha_img=BACKGROUND_ALPHA
        )

        # C. 保存
        save_file = os.path.join(PLOT_DIR, f'{pathway}_vector_field_large.pdf')
        plt.savefig(save_file, bbox_inches='tight', dpi=300)  # 提高DPI确保清晰
        plt.close()

        print(f"       Saved: {save_file}")

    except Exception as e:
        print(f"       [Error] Failed to process {pathway}: {e}")
        # 如果是因为 alpha_img 参数报错，去掉那个参数再试一次

print("Batch processing completed.")

# -------------------------------------------------------------------------
# [Module 6] 保存结果
# -------------------------------------------------------------------------
print("[Module 6] Saving Results...")

# 由于 COMMOT 在 obsm 中存储了大量稀疏矩阵，h5ad 会变得很大
# 建议只保存必要部分，或者压缩保存
save_path = os.path.join(OUT_DIR, "commot_results.h5ad")
adata.write_h5ad(save_path, compression="gzip")

print("=" * 60)
print(f"COMMOT Analysis Completed.")
print(f"Results saved to: {OUT_DIR}")
print("=" * 60)