# -*- coding: utf-8 -*-
"""
===========================================================================
文件名称: st_bbknn_integration_final.py
作    者: Kuroneko
创建日期: 2026-02-01
版    本: V1.0

功能描述:
    [多样本 BBKNN 去批次整合分析流程]
    针对 F:/ST/code/02/different_data/10x.s/ 目录下的 4 个样本进行整合分析。

    核心步骤:
    1. 数据加载: 读取 4 个不同来源的 10x 数据 (CAFs.1, CAFs.2, dapi1, dapi2)。
       * 包含对旧版 10x 格式 (features.tsv 仅两列) 的兼容性修复。
    2. 元数据构建:
       * Batch: 标记具体样本来源 (如 CAFs.1)。
       * Condition: 标记生物学分组 (CAFs 组 vs DAPI 组)。
    3. 整合前评估: 基于原始 PCA 绘制 UMAP，展示原始数据的批次效应。
    4. BBKNN 整合: 使用 Batch Balanced KNN 算法消除技术批次效应。
    5. 整合后评估: 绘制修正后的 UMAP，按 Batch 和 Condition 上色对比。

输入文件 (Input):
    - F:/ST/code/02/different_data/10x.s/CAFs.1
    - F:/ST/code/02/different_data/10x.s/CAFs.2
    - F:/ST/code/02/different_data/10x.s/dapi1
    - F:/ST/code/02/different_data/10x.s/dapi2

输出文件 (Output):
    - F:/ST/code/results/integrated_bbknn_final.h5ad
    - F:/ST/code/results/figures_bbknn/ (包含整合前后的对比图)
===========================================================================
"""

import os
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================================
# [Module 0] 环境初始化
# =========================================================================
print("[Module 0] 初始化系统环境...")

BASE_DIR = r"F:\ST\code"
DATA_DIR = os.path.join(BASE_DIR, "02", "different_data", "10x.s")
RESULT_DIR = os.path.join(BASE_DIR, "results")
FIGURE_DIR = os.path.join(RESULT_DIR, "figures_bbknn")

# 创建输出目录
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)
if not os.path.exists(FIGURE_DIR):
    os.makedirs(FIGURE_DIR)

# 设置绘图参数
sc.settings.verbosity = 3
sc.settings.figdir = FIGURE_DIR
sc.set_figure_params(dpi=150, facecolor="white", vector_friendly=True)

print(f"数据输入目录: {DATA_DIR}")
print(f"图表输出目录: {FIGURE_DIR}")

# =========================================================================
# [Module A] 数据加载与元数据构建
# =========================================================================
print("\n[Module A] 开始加载样本并构建分组信息...")

# 定义样本与分组对应关系
# Key: 文件夹名 (Batch)
# Value: 生物学分组 (Condition)
sample_map = {
    "CAFs.1": "CAFs",
    "CAFs.2": "CAFs",
    "dapi1": "DAPI",
    "dapi2": "DAPI"
}

adatas_list = []

for sample_name, condition in sample_map.items():
    sample_path = os.path.join(DATA_DIR, sample_name)

    if not os.path.exists(sample_path):
        print(f"[Warning] 样本路径不存在，跳过: {sample_path}")
        continue

    print(f"--> 读取样本: {sample_name} (Group: {condition})")

    # 尝试读取数据 (包含兼容模式)
    try:
        # 优先尝试标准读取
        ad = sc.read_10x_mtx(sample_path, var_names='gene_symbols', cache=True)
    except KeyError:
        print(f"    [Info] 检测到旧版格式 (KeyError: 2)，切换至兼容模式读取...")
        try:
            mat_file = os.path.join(sample_path, "matrix.mtx.gz")
            feat_file = os.path.join(sample_path, "features.tsv.gz")
            bar_file = os.path.join(sample_path, "barcodes.tsv.gz")

            # 手动读取并转置 (Matrix Market 格式通常是 Genes x Cells)
            ad = sc.read_mtx(mat_file).T
            features = pd.read_csv(feat_file, header=None, sep='\t')
            barcodes = pd.read_csv(bar_file, header=None, sep='\t')

            # 手动赋值
            ad.var_names = features[1].values  # 基因名
            ad.var['gene_ids'] = features[0].values  # 基因ID
            ad.obs_names = barcodes[0].values  # 细胞条码
        except Exception as e:
            print(f"    [Error] 兼容模式读取也失败了: {e}")
            continue
    except Exception as e:
        print(f"    [Error] 未知读取错误: {e}")
        continue

    # 基础处理
    ad.var_names_make_unique()

    # [关键步骤] 写入元数据
    ad.obs['batch'] = sample_name  # 技术批次: CAFs.1, dapi1...
    ad.obs['condition'] = condition  # 生物分组: CAFs, DAPI

    # 单独保存 (物理备份)
    single_save = os.path.join(RESULT_DIR, f"{sample_name}.h5ad")
    ad.write(single_save, compression="gzip")
    print(f"    已独立存档: {single_save}")

    adatas_list.append(ad)

if not adatas_list:
    print("[Error] 没有成功加载任何数据，程序终止。")
    exit()

# =========================================================================
# [Module B] 数据整合与预处理
# =========================================================================
print("\n[Module B] 合并数据并执行预处理...")

# 1. 拼接
adata = sc.concat(adatas_list, label="batch_source", keys=sample_map.keys(), join="outer")
# 确保 batch 列是 categorical 类型 (绘图需要)
adata.obs['batch'] = adata.obs['batch'].astype('category')
adata.obs['condition'] = adata.obs['condition'].astype('category')

print(f"    合并后数据规模: {adata.shape}")

# 2. 质控与过滤
print("--> 执行质控与标准化...")
adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

# 简单过滤
sc.pp.filter_cells(adata, min_counts=200)
sc.pp.filter_genes(adata, min_cells=3)
adata = adata[adata.obs.pct_counts_mt < 20].copy()

# 标准化
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# 3. PCA
print("--> 计算高变基因与 PCA...")
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
adata = adata[:, adata.var.highly_variable]
sc.pp.pca(adata, svd_solver='arpack')

# =========================================================================
# [Module C] 整合前评估 (Before Integration)
# =========================================================================
print("\n[Module C] 评估整合前的批次效应 (Raw PCA)...")

# 使用原始 PCA 计算邻居图
# 这里的 neighbors 是基于没有去批次的 PCA 算的
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=40)
sc.tl.umap(adata)

# 绘图 1: 整合前的情况
print("--> 绘制整合前 UMAP...")
sc.pl.umap(adata, color=['batch', 'condition'], wspace=0.4,
           title=['Batch Effect (Before Integration)', 'Condition (Before Integration)'],
           show=False, save="_Before_BBKNN.png")

# =========================================================================
# [Module D] BBKNN 整合与效果评估 (After Integration)
# =========================================================================
print("\n[Module D] 运行 BBKNN 去批次并评估效果...")

try:
    # 运行 BBKNN
    # 注意: BBKNN 直接修改 adata.obsp['distances'] (邻居图)
    # 它不需要修改 PCA 坐标，而是强制让不同 batch 的细胞互为邻居
    sc.external.pp.bbknn(adata, batch_key='batch', neighbors_within_batch=3)
    print("    BBKNN 计算完成 (邻居图已重构)")

    # 重新计算 UMAP (基于 BBKNN 修正后的邻居图)
    print("--> 基于 BBKNN 结果计算 UMAP...")
    sc.tl.umap(adata)

    # 绘图 2: 整合后的情况
    print("--> 绘制整合后 UMAP...")
    sc.pl.umap(adata, color=['batch', 'condition'], wspace=0.4,
               title=['Batch Effect (After BBKNN)', 'Condition (After BBKNN)'],
               show=False, save="_After_BBKNN.png")

    # 保存最终结果
    save_path = os.path.join(RESULT_DIR, "integrated_bbknn_final.h5ad")
    adata.write(save_path, compression="gzip")
    print(f"\n[Success] 最终结果已保存: {save_path}")

except ImportError:
    print("[Error] 缺少 bbknn 库。请在终端运行: pip install bbknn")
except Exception as e:
    print(f"[Error] BBKNN 运行失败: {e}")

print("-" * 60)
print("分析流程执行完毕。请查看 figures_bbknn 文件夹下的对比图。")
print("-" * 60)