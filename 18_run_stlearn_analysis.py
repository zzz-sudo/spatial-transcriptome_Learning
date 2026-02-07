# ===========================================================================
# File Name: 18_run_stlearn_analysis.py
# Author: Kuroneko
# Version: V1.1 (Fix Image Loading Error)
#
# Description:
#     [stLearn 空间转录组综合分析 - 教学版]
#
#     [修复记录 V1.1]:
#     - 移除了过时的 st.pp.loading_image() 函数，解决了 AttributeError。
#       (st.Read10X 已经自动加载了图像，无需重复加载)。
#     - 移除了 Module 3 中重复的 normalization/log1p 步骤，消除了 Warning。
#
#     [核心功能]:
#     1. SME Clustering: 融合 "形态+基因+空间" 的聚类。
#     2. CCI: 细胞互作分析。
#     3. Tiling: 自动生成切片小图。
#
# Input Files:
#     - Data Root: F:/ST/code/data/breast_cancer/
#
# Output Files (in F:/ST/code/results/stlearn_output/):
#     - tiles/: 切片图片
#     - plots/: 结果图
#     - stlearn_results.h5ad: 最终数据
# ===========================================================================

import sys
import os
import gc
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import stlearn as st

# 忽略部分警告
warnings.filterwarnings("ignore")

# 设置绘图参数
sc.settings.set_figure_params(dpi=120, frameon=False, facecolor='white')
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Arial'

# -------------------------------------------------------------------------
# [Module 1] 环境与路径初始化
# -------------------------------------------------------------------------
print("[Module 1] Initializing Environment...")

# 路径设置
BASE_DIR = r"F:/ST/code/"
DATA_DIR = os.path.join(BASE_DIR, "data", "breast_cancer")
OUT_DIR = os.path.join(BASE_DIR, "results", "stlearn_output")
TILES_DIR = os.path.join(OUT_DIR, "generated_tiles")
PLOT_DIR = os.path.join(OUT_DIR, "plots")

for d in [OUT_DIR, TILES_DIR, PLOT_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

print(f"    Input Data: {DATA_DIR}")
print(f"    Output Dir: {OUT_DIR}")
print(f"    Tiles Save Dir: {TILES_DIR}")

# -------------------------------------------------------------------------
# [Module 2] 数据加载
# -------------------------------------------------------------------------
print("\n[Module 2] Loading Data (Visium)...")

# stLearn 读取 10x 数据的标准方式
data = st.Read10X(DATA_DIR)

# 过滤低质量细胞和基因 (常规 QC)
sc.pp.filter_genes(data, min_cells=3)
sc.pp.normalize_total(data, target_sum=1e4)
sc.pp.log1p(data)

print(f"    Data Loaded: {data.shape[0]} spots x {data.shape[1]} genes")

# -------------------------------------------------------------------------
# [Module 3] 图像切片与特征提取
# -------------------------------------------------------------------------
print("\n[Module 3] Tiling & Feature Extraction (The Magic of Images)...")

# [Step 3.1] 切片 (Tiling)
print(f"    Generating tiles from the high-res image (Saving to {TILES_DIR})...")

try:
    st.pp.tiling(data, out_path=TILES_DIR, crop_size=299, target_size=299, verbose=True)
    print("    [Check] You can now go to the 'generated_tiles' folder to see the images!")
except Exception as e:
    print(f"    [Error] Tiling failed: {e}")

# [Step 3.2] 提取深度学习特征
print("    Extracting morphological features using ResNet50 (Use GPU if available)...")

try:
    st.pp.extract_feature(data)
except Exception as e:
    print(f"    [Warning] Feature extraction failed: {e}")
    print("    Skipping SME clustering relying on image features.")

# -------------------------------------------------------------------------
# [Module 4] SME 聚类 (结合 形态 + 基因 + 空间)
# -------------------------------------------------------------------------
print("\n[Module 4] SME Clustering (Integrating Morphology & Gene Expression)...")

# 1. 运行 PCA (针对基因)
# [FIX V1.2] 使用 n_comps 替代 n_pcs
try:
    st.em.run_pca(data, n_comps=50)
except TypeError:
    st.em.run_pca(data, n_pcs=50)  # 兼容旧版本

# 2. 融合特征 (SME 标准化)
if 'X_tile_feature' in data.obsm.keys():
    print("    Applying SME normalization (Morphology weights)...")
    st.spatial.SME.SME_normalize(data, use_data="raw", weights="physical_distance")

    # 3. 再次运行 PCA (针对融合后的数据)
    data.X = data.obsm['raw_SME_normalized']
    try:
        st.em.run_pca(data, n_comps=50)
    except TypeError:
        st.em.run_pca(data, n_pcs=50)
else:
    print("    [Note] Image features not found. Using standard gene expression clustering.")

# 4. 聚类 (K-means)
n_clusters = 10
print(f"    Clustering into {n_clusters} groups...")
st.tl.clustering.kmeans(data, n_clusters=n_clusters, use_data="X_pca", key_added="X_pca_kmeans")

# 5. 绘图：SME 聚类结果
plt.figure(figsize=(8, 8))
st.pl.cluster_plot(data, use_label="X_pca_kmeans")
plt.title("SME Clustering (Morphology + Gene)")
plt.savefig(os.path.join(PLOT_DIR, "01_SME_clusters.png"), bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------------
# [Module 5] 细胞互作分析 (CCI - Cell-Cell Interaction)
# -------------------------------------------------------------------------
print("\n[Module 5] Cell-Cell Interaction (CCI) Analysis...")


# [FIX V1.3] 智能加载数据库函数
def safe_load_lr_db(data, species='human', db_name='connecthed'):
    # 尝试 1: 标准路径
    if hasattr(st.tl.cci, 'load_lrcell_db'):
        print("    -> Using st.tl.cci.load_lrcell_db")
        st.tl.cci.load_lrcell_db(data, species=species, db_name=db_name)
        return True

    # 尝试 2: 备用名称
    if hasattr(st.tl.cci, 'load_lrcell'):
        print("    -> Using st.tl.cci.load_lrcell")
        st.tl.cci.load_lrcell(data, species=species, db_name=db_name)
        return True

    # 尝试 3: 底层路径 (Deep Import)
    try:
        from stlearn.tools.cci.base import load_lrcell_db
        print("    -> Using stlearn.tools.cci.base.load_lrcell_db")
        load_lrcell_db(data, species=species, db_name=db_name)
        return True
    except ImportError:
        pass

    return False


# 1. 加载数据库
print("    Loading LR database...")
db_loaded = safe_load_lr_db(data, species='human', db_name='connecthed')

if db_loaded:
    # 2. 运行排列测试
    print("    Running CCI permutation test (this might take a minute)...")
    try:
        st.tl.cci.run(data, data.uns['lrcell_db'], min_spots=20, distance=None, n_pairs=1000, n_cpus=4)

        # -------------------------------------------------------------------------
        # [Module 6] 高级可视化
        # -------------------------------------------------------------------------
        print("\n[Module 6] Generating Advanced Plots...")

        # [图表 A] 互作气泡图
        print("    Plotting LR interaction bubble plot...")
        try:
            st.pl.lr_plot(data, n_pairs=15, show=False)
            plt.savefig(os.path.join(PLOT_DIR, "02_LR_pairs_bubble.png"), bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"    Bubble plot failed: {e}")

        # [图表 B] 具体的受体-配体空间分布图
        try:
            if 'lr_summary' in data.uns and not data.uns['lr_summary'].empty:
                best_lr_pair = data.uns['lr_summary'].index[0]  # 获取排名第一的互作对
                print(f"    Visualizing top LR pair: {best_lr_pair}")

                plt.figure(figsize=(10, 5))
                st.pl.lr_result_plot(data, use_result=best_lr_pair, show_color_bar=True, show_spot=True)
                plt.savefig(os.path.join(PLOT_DIR, f"03_Spatial_LR_{best_lr_pair}.png"), bbox_inches='tight')
                plt.close()
            else:
                print("    No significant LR pairs found to plot.")
        except Exception as e:
            print(f"    LR spatial plot failed: {e}")

    except Exception as e:
        print(f"    CCI Run failed: {e}")
else:
    print("    [Error] Could not find function to load LR database. Skipping CCI analysis.")

# [图表 C] 基因/特征 融合展示 (独立于 CCI)
target_gene = 'FASN'
if target_gene in data.var_names:
    print(f"    Plotting gene expression: {target_gene}")
    st.pl.gene_plot(data, gene_symbols=target_gene)
    plt.savefig(os.path.join(PLOT_DIR, f"04_Gene_{target_gene}.png"), bbox_inches='tight')
    plt.close()

# -------------------------------------------------------------------------
# [Module 7] 保存结果
# -------------------------------------------------------------------------
print("\n[Module 7] Saving Results...")

save_path = os.path.join(OUT_DIR, "stlearn_results.h5ad")
data.write_h5ad(save_path)

print("=" * 60)
print(f"stLearn Analysis Completed.")
print(f"1. Generated Tiles are in: {TILES_DIR}")
print(f"2. Plots are in: {PLOT_DIR}")

print("=" * 60)
