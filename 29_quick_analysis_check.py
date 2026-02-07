# ===========================================================================
# Project: Spatial Transcriptomics - QC & Visualization Check
# File Name: 36_quick_analysis_check.py
# Author: Kuroneko
# Current Path: F:\ST\code
#
# ---------------------------------------------------------------------------
# [1. 脚本功能]
# ---------------------------------------------------------------------------
# 本脚本用于快速验证上一步 (StarDist/Cellpose) 生成的单细胞数据质量。
# 核心检查点：
# 1. 细胞数量是否合理 (预期 10万+)。
# 2. 空间坐标 (obsm['spatial']) 是否存在且范围正确。
# 3. 聚类结果是否在空间上有明显的生物学分布 (而不是随机杂乱)。
#
# ---------------------------------------------------------------------------
# [2. 输入文件]
# ---------------------------------------------------------------------------
# - 单细胞级空间数据: F:\ST\code\results\seg\adata_cell_level.h5ad
#
# ---------------------------------------------------------------------------
# [3. 输出文件]
# ---------------------------------------------------------------------------
# - 质控对比图: F:\ST\code\results\seg\36_qc_spatial_check.png
# ===========================================================================

import os
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# [Module 1] 环境初始化
# ---------------------------------------------------------------------------
print("[Module 1] Initializing...")

DATA_PATH = r"F:\ST\code\results\seg\adata_cell_level.h5ad"
OUT_DIR = r"F:\ST\code\results\seg"

if not os.path.exists(DATA_PATH):
    print(f"Error: 找不到数据文件 {DATA_PATH}")
    print("请先运行 29_cell_segmentation_and_binning.py 生成数据。")
    exit(1)

# ---------------------------------------------------------------------------
# [Module 2] 加载数据与坐标检查
# ---------------------------------------------------------------------------
print("\n[Module 2] Loading Data...")
adata = sc.read_h5ad(DATA_PATH)

print(f"    Data Shape: {adata.n_obs} cells x {adata.n_vars} genes")

# [关键检查] 验证空间坐标
if 'spatial' in adata.obsm:
    print("    [Pass] Spatial coordinates found in obsm['spatial'].")
    coords = adata.obsm['spatial']
    print(f"           Range X: {coords[:, 0].min():.1f} - {coords[:, 0].max():.1f}")
    print(f"           Range Y: {coords[:, 1].min():.1f} - {coords[:, 1].max():.1f}")
else:
    print("    [Fail] Warning: 'spatial' key missing in obsm!")
    print("           无法绘制空间图。请确保你运行的是'修复版'的脚本 29。")

# ---------------------------------------------------------------------------
# [Module 3] 快速聚类分析
# ---------------------------------------------------------------------------
print("\n[Module 3] Running Quick Clustering...")

# 简单的预处理
sc.pp.filter_cells(adata, min_genes=50) # 过滤掉只有极少基因的空细胞
print(f"    Cells after filtering: {adata.n_obs}")

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# 降维聚类
print("    Running PCA & UMAP...")
sc.pp.pca(adata, n_comps=30)
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5, key_added='leiden')

# ---------------------------------------------------------------------------
# [Module 4] 可视化对比
# ---------------------------------------------------------------------------
print("\n[Module 4] Plotting...")

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# 图1: UMAP 聚类
sc.pl.umap(adata, color='leiden', ax=axs[0], show=False, title="UMAP Clustering", legend_loc='on data')

# 图2: 空间分布
if 'spatial' in adata.obsm:
    # spot_size 设置小一点，防止点太密糊在一起
    sc.pl.embedding(adata, basis='spatial', color='leiden', ax=axs[1], show=False,
                    s=1.5, title="Spatial Map (Reconstructed)")
    axs[1].invert_yaxis() # 显微镜坐标通常 Y 轴向下，可能需要翻转
else:
    axs[1].text(0.5, 0.5, "No Spatial Coords Found", ha='center', fontsize=14)

plt.tight_layout()
save_path = os.path.join(OUT_DIR, "36_qc_spatial_check.png")
plt.savefig(save_path, dpi=300)
plt.close()

print("="*60)
print(f"Check Completed.")
print(f"QC Plot saved to: {save_path}")
print("Please open the PNG image to verify spatial structure.")

print("="*60)
