# -*- coding: utf-8 -*-
"""
---------------------------------------------------------------------------
文件名称: st_multisample_analysis.py
作    者: Kuroneko
创建日期: 2026-02-01
功能描述:
    多样本空间转录组整合分析。
    1. 读取两个不同的空间转录组切片 (Anterior 和 Posterior)。
    2. 对每个样本进行独立的预处理 (归一化、高变基因)。
    3. 数据拼接 (Concatenate) 与整合。
    4. 降维聚类 (PCA, UMAP, Leiden)。
    5. 可视化：对比不同样本在 UMAP 和 空间上的分布。

输入文件:
    F:/ST/code/02/adata_spatial_anterior.h5ad (前脑切片)
    F:/ST/code/02/adata_spatial_posterior.h5ad (后脑切片)

输出文件:
    F:/ST/code/results/ (包含整合后的 UMAP 图和空间分布图)


adata_spatial_anterior.h5ad 和 adata_spatial_posterior.h5ad：
含义：这是两张小鼠大脑的切片数据。Anterior 是前脑切片，Posterior 是后脑切片。
来源：这是 10x Genomics 官方提供的公共演示数据，或者来自 Sagittal Mouse Brain 的公共数据集。
---------------------------------------------------------------------------
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ================= 1. 环境与路径设置 =================
print("Step 1: 环境配置...")

# 基础路径
base_dir = r"F:\ST\code"
# 输入路径 (根据你的 dir /s 结果，数据直接在 02 目录下)
input_dir = os.path.join(base_dir, "02")
# 输出路径
result_dir = os.path.join(base_dir, "results")
figure_dir = os.path.join(result_dir, "figures")

# 自动创建输出目录
for path in [result_dir, figure_dir]:
    if not os.path.exists(path):
        os.makedirs(path)

# Scanpy 绘图设置
sc.settings.verbosity = 3
sc.settings.figdir = figure_dir
sc.set_figure_params(dpi=150, facecolor="white", figsize=(8, 8))

print(f"输入目录: {input_dir}")
print(f"输出目录: {result_dir}")

# ================= 2. 读取多样本数据 =================
print("\nStep 2: 读取两个切片数据...")

# 路径定义
path_ant = os.path.join(input_dir, "adata_spatial_anterior.h5ad")
path_post = os.path.join(input_dir, "adata_spatial_posterior.h5ad")

try:
    # 读取 h5ad 文件
    adata_ant = sc.read_h5ad(path_ant)
    adata_post = sc.read_h5ad(path_post)

    # 确保基因名唯一
    adata_ant.var_names_make_unique()
    adata_post.var_names_make_unique()

    # [关键] 给每个样本打上标签，方便合并后区分
    adata_ant.obs['batch'] = 'Anterior'  # 前脑
    adata_post.obs['batch'] = 'Posterior'  # 后脑

    print(f"读取成功!")
    print(f"Anterior 样本: {adata_ant.n_obs} spots")
    print(f"Posterior 样本: {adata_post.n_obs} spots")

except Exception as e:
    print(f"读取失败: {e}")
    print("请检查 F:\\ST\\code\\02 目录下是否存在 adata_spatial_anterior.h5ad 等文件。")
    exit()

# ================= 3. 独立预处理 (QC & Norm) =================
print("\nStep 3: 对每个样本进行预处理...")


# 定义一个处理函数，避免写两遍代码
def preprocess_sample(adata, sample_name):
    print(f"正在处理: {sample_name}...")

    # 1. 计算 QC 指标
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

    # 2. 简单的过滤 (根据经验值，你也可以画图看)
    # 这里为了保证代码跑通，使用宽松阈值
    sc.pp.filter_cells(adata, min_counts=500)
    adata = adata[adata.obs["pct_counts_mt"] < 25].copy()

    # 3. 归一化与对数化
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # 4. 寻找高变基因 (HVG)
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)
    return adata


# 分别处理两个样本
adata_ant = preprocess_sample(adata_ant, "Anterior")
adata_post = preprocess_sample(adata_post, "Posterior")

# ================= 4. 数据整合 (Concatenate) =================
print("\nStep 4: 合并样本 (Concatenate)...")

# 使用 scanpy.concat 合并
# label="batch" 会根据我们在 Step 2 设置的 batch 列来标记来源
# join="outer" 取并集，防止丢失基因
adata_all = sc.concat([adata_ant, adata_post], label="batch", keys=["Anterior", "Posterior"])

# [重点] 恢复空间信息 (Spatial info)
# concat 默认会丢掉 uns['spatial'] 里的图片信息，必须手动加回来
adata_all.uns['spatial'] = {}
# 把 Anterior 的图片加进去
if 'spatial' in adata_ant.uns:
    adata_all.uns['spatial'].update(adata_ant.uns['spatial'])
# 把 Posterior 的图片加进去
if 'spatial' in adata_post.uns:
    adata_all.uns['spatial'].update(adata_post.uns['spatial'])

print(f"合并完成。总数据形状: {adata_all.shape}")
print(f"Batch 分布:\n{adata_all.obs['batch'].value_counts()}")

# ================= 5. 降维与聚类 =================
print("\nStep 5: 降维与聚类...")

# 这里的 .X 已经是归一化后的数据
sc.pp.pca(adata_all, n_comps=50)

# 计算邻近图
sc.pp.neighbors(adata_all, n_neighbors=15, n_pcs=40)

# 计算 UMAP
sc.tl.umap(adata_all)

# 聚类 (Leiden)
sc.tl.leiden(adata_all, resolution=0.5, key_added="clusters")

# ================= 6. 结果可视化 =================
print("\nStep 6: 绘制多样本对比图...")

# 6.1 UMAP - 查看是否存在批次效应
# 如果两个颜色的点完全分开，说明有批次效应；如果混在一起，说明整合得好
plt.rcParams["figure.figsize"] = (5, 5)
sc.pl.umap(adata_all, color=["batch", "clusters"], wspace=0.4,
           title=["Batch (Sample ID)", "Clusters"],
           show=False, save="_multisample_umap.png")

# 6.2 空间分布图 - 分别画两个样本
# 这是一个高级技巧：从大文件中切分出小文件来画图
print("正在绘制空间分布图...")

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# 提取 Anterior 的数据并画图
# library_id 是存储图片信息的关键 key，需要从 uns['spatial'] 里找
try:
    # 获取第一个样本的 library_id
    lib_id_ant = list(adata_ant.uns['spatial'].keys())[0]
    # 提取子集
    subset_ant = adata_all[adata_all.obs['batch'] == 'Anterior']
    sc.pl.spatial(subset_ant, library_id=lib_id_ant, color="clusters",
                  img_key="hires", size=1.5, title="Anterior Slice",
                  ax=axs[0], show=False)
except:
    axs[0].text(0.5, 0.5, "Anterior Plot Error", ha='center')

# 提取 Posterior 的数据并画图
try:
    lib_id_post = list(adata_post.uns['spatial'].keys())[0]
    subset_post = adata_all[adata_all.obs['batch'] == 'Posterior']
    sc.pl.spatial(subset_post, library_id=lib_id_post, color="clusters",
                  img_key="hires", size=1.5, title="Posterior Slice",
                  ax=axs[1], show=False)
except:
    axs[1].text(0.5, 0.5, "Posterior Plot Error", ha='center')

plt.tight_layout()
plt.savefig(os.path.join(figure_dir, "multisample_spatial_clusters.png"))
plt.close()

# ================= 7. 保存结果 =================
print("\nStep 7: 保存整合后的数据...")
save_path = os.path.join(result_dir, "integrated_analysis.h5ad")
adata_all.write(save_path)
print(f"结果已保存至: {save_path}")

print("\n=== 多样本分析结束 ===")