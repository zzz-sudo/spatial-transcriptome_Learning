# -*- coding: utf-8 -*-
"""
===========================================================================
文件名称: st_pnas_merfish_analysis.py
作    者: Kuroneko
创建日期: 2026-02-01
版    本: V1.0

功能描述:
    [MERFISH 空间转录组数据构建与分析]
    针对 PNAS 文献中的非标准格式数据进行手动组装与分析。

    核心流程:
    1. 数据组装:
       - 读取表达量矩阵 (.csv) 并转置为 (Cells x Genes) 格式。
       - 读取空间坐标文件 (.xlsx) 提取 X,Y 坐标。
       - 基于细胞 ID (Index) 对齐两个文件，构建 AnnData 对象。
    2. 空间注册: 将坐标信息写入 .obsm['spatial']，赋予数据空间属性。
    3. 标准分析: 执行质控、归一化、对数变换、PCA 降维。
    4. 聚类降维: 计算邻居图、UMAP 投影和 Leiden 聚类。
    5. 可视化: 绘制 UMAP 分布图与空间位置分布图 (Spatial Plot)。

输入文件 (Input):
    - F:/ST/code/02/pnas/pnas.1912459116.sd12.csv  (表达矩阵)
    - F:/ST/code/02/pnas/pnas.1912459116.sd15.xlsx (坐标信息)

输出文件 (Output):
    - F:/ST/code/results/pnas_merfish_final.h5ad
    - F:/ST/code/results/figures_pnas/ (包含空间分布图等)
===========================================================================
"""

import os
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================================
# [Module 0] 环境初始化
# =========================================================================
print("[Module 0] 初始化系统环境...")

BASE_DIR = r"F:\ST\code"
DATA_DIR = os.path.join(BASE_DIR, "02", "pnas")
RESULT_DIR = os.path.join(BASE_DIR, "results")
FIGURE_DIR = os.path.join(RESULT_DIR, "figures_pnas")

# 创建输出目录
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)
if not os.path.exists(FIGURE_DIR):
    os.makedirs(FIGURE_DIR)

# 设置绘图参数
sc.settings.verbosity = 3
sc.settings.figdir = FIGURE_DIR
sc.set_figure_params(dpi=150, facecolor="white", vector_friendly=True)

print(f"数据目录: {DATA_DIR}")
print(f"结果目录: {RESULT_DIR}")

# =========================================================================
# [Module A] 数据加载与对象组装
# =========================================================================
print("\n[Module A] 加载表达矩阵与空间坐标...")

file_counts = os.path.join(DATA_DIR, "pnas.1912459116.sd12.csv")
file_coords = os.path.join(DATA_DIR, "pnas.1912459116.sd15.xlsx")

if not (os.path.exists(file_counts) and os.path.exists(file_coords)):
    print(f"[Error] 数据文件缺失，请检查路径。")
    print(f"    期望路径: {file_counts}")
    print(f"    期望路径: {file_coords}")
    exit()

# 1. 读取表达矩阵
print("--> 读取表达矩阵 (CSV)...")
# 注意: 原始数据通常行是基因，列是细胞，所以需要 .T (转置)
try:
    counts_df = pd.read_csv(file_counts, index_col=0).T
    print(f"    矩阵维度 (Cells x Genes): {counts_df.shape}")
except Exception as e:
    print(f"    [Error] CSV 读取失败: {e}")
    exit()

# 2. 读取坐标文件
print("--> 读取空间坐标 (Excel)...")
try:
    coords_df = pd.read_excel(file_coords, index_col=0)
    print(f"    坐标维度: {coords_df.shape}")
    # 通常坐标文件包含 'x' 和 'y' 列，或者类似的数值列
    print(f"    坐标列名: {coords_df.columns.tolist()}")
except ImportError:
    print("    [Error] 缺少 openpyxl 库。请运行: pip install openpyxl")
    exit()
except Exception as e:
    print(f"    [Error] Excel 读取失败: {e}")
    exit()

# 3. 对齐数据 (取交集)
print("--> 对齐细胞索引...")
common_cells = counts_df.index.intersection(coords_df.index)
print(f"    共有细胞数量: {len(common_cells)}")

if len(common_cells) == 0:
    print("[Error] 表达矩阵与坐标文件的细胞ID没有重叠，无法合并！")
    exit()

counts_df = counts_df.loc[common_cells]
coords_df = coords_df.loc[common_cells]

# 4. 构建 AnnData
print("--> 构建 AnnData 空间对象...")
adata = sc.AnnData(counts_df)

# [关键步骤] 将坐标写入 obsm['spatial']
# Scanpy 的空间绘图函数默认从这里读取坐标
# 确保坐标是 numpy array 格式
adata.obsm['spatial'] = coords_df.to_numpy()

# 记录一下原始坐标列名到 uns 中备查
adata.uns['spatial_cols'] = coords_df.columns.tolist()

print(f"    对象构建完成: {adata}")

# =========================================================================
# [Module B] 数据预处理
# =========================================================================
print("\n[Module B] 执行标准预处理流程...")

# 1. 基础过滤
sc.pp.filter_cells(adata, min_counts=10)  # 过滤掉极低表达的细胞
sc.pp.filter_genes(adata, min_cells=3)

# 2. 归一化与对数变换
print("--> 归一化与 Log1p...")
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# 3. PCA 降维
print("--> 计算 PCA...")
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
adata.raw = adata  # 备份原始数据
adata = adata[:, adata.var.highly_variable]
sc.pp.scale(adata, max_value=10)
sc.pp.pca(adata, svd_solver='arpack')

# =========================================================================
# [Module C] 聚类与流形学习
# =========================================================================
print("\n[Module C] 计算邻居图与聚类...")

# 1. 计算邻居 (基于表达量相似性)
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=40)

# 2. UMAP 降维 (用于展示表达模式的相似性)
sc.tl.umap(adata)

# 3. Leiden 聚类
sc.tl.leiden(adata, resolution=0.5, key_added='leiden')

# =========================================================================
# [Module D] 可视化 (UMAP vs Spatial)
# =========================================================================
print("\n[Module D] 生成分析图表...")

# 图 1: UMAP 聚类图 (展示转录组相似性)
print("--> 绘制 UMAP 聚类图...")
sc.pl.umap(adata, color=['leiden'], title='Transcriptomic Clusters (UMAP)',
           show=False, save='_UMAP_Clusters.png')

# 图 2: 空间分布图 (展示细胞在组织中的真实位置)
# 使用 embedding 函数，basis='spatial' 指定使用物理坐标
print("--> 绘制空间分布图...")
sc.pl.embedding(adata, basis='spatial', color=['leiden'],
                title='Spatial Distribution of Clusters',
                size=30,  # MERFISH 点通常较多，调节点大小
                show=False, save='_Spatial_Clusters.png')

# 图 3: 对比图 (左边 UMAP，右边 Spatial)
print("--> 绘制对比拼图...")
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# 左图: UMAP
sc.pl.umap(adata, color='leiden', ax=axs[0], show=False, title='UMAP Space', legend_loc='on data')
# 右图: Spatial
sc.pl.embedding(adata, basis='spatial', color='leiden', ax=axs[1], show=False, title='Physical Space')

plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "Comparison_UMAP_vs_Spatial.png"))
plt.close()

# =========================================================================
# [Module E] 结果保存
# =========================================================================
print("\n[Module E] 保存分析结果...")

save_path = os.path.join(RESULT_DIR, "pnas_merfish_final.h5ad")
adata.write(save_path, compression="gzip")

print("-" * 60)
print("PNAS MERFISH 数据构建与分析完成。")
print(f"结果文件: {save_path}")
print(f"图表目录: {FIGURE_DIR}")
print("-" * 60)