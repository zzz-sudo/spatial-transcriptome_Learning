# ===========================================================================
# File Name: 17_run_tangram_analysis_fixed_v1.3.py
# Author: Kuroneko
# Date: 2026-02-03
# Version: V1.3 (Fix Histogram Dimensionality Error)
#
# Description:
#     [Tangram 单细胞空间映射 - 增强版修复]
#
#     修复记录 (V1.3):
#     - [Critical Fix]: 在绘制映射概率直方图时，增加了 .flatten() 操作。
#       解决了因输入二维矩阵导致的 "ValueError: ... but 4035 datasets ... provided"。
#
#     功能包含:
#     1. [Mapping]: 单细胞到空间的映射。
#     2. [Imputation]: 全转录组基因插补。
#     3. [Visualization]: 批量导出 CSV 和 图片。
#
# Input Files:
#     - Spatial: F:/ST/code/data/V1_Human_Lymph_Node/
#     - Reference: F:/ST/code/03/sc.h5ad
#
# Output Files (in F:/ST/code/results/tangram_output/):
#     - cell_type_predictions.csv
#     - imputed_genes_subset.csv
#     - plots/
# ===========================================================================

import sys
import os
import gc
import torch
import tangram as tg
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import issparse
import warnings

# 忽略不必要的警告，保持日志整洁
warnings.filterwarnings("ignore")

# -------------------------------------------------------------------------
# [全局设置]
# -------------------------------------------------------------------------
# 设置绘图参数：高DPI，白色背景，Arial字体
sc.settings.set_figure_params(dpi=120, frameon=False, facecolor='white')
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Arial'


def clean_gpu():
    """清理 GPU 显存，防止 OOM 错误"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# 检测硬件环境
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Hardware Check: Using device {device}")

# -------------------------------------------------------------------------
# [Module 1] 路径定义与环境初始化
# -------------------------------------------------------------------------
print("\n[Module 1] Initializing Paths...")

ROOT_DIR = r"F:/ST/code"
# 确保这里指向你的 Visium 数据文件夹
DATA_DIR = os.path.join(ROOT_DIR, "data", "V1_Human_Lymph_Node")
# 确保这里指向你的单细胞 h5ad 文件
SC_REF_PATH = os.path.join(ROOT_DIR, "03", "sc.h5ad")

# 输出目录结构
OUT_DIR = os.path.join(ROOT_DIR, "results", "tangram_output")
PLOT_DIR = os.path.join(OUT_DIR, "plots")
PLOT_CT_DIR = os.path.join(PLOT_DIR, "cell_types")  # 存放细胞类型分布图
PLOT_QC_DIR = os.path.join(PLOT_DIR, "qc")  # 存放质量控制图

# 自动创建目录
for path in [OUT_DIR, PLOT_DIR, PLOT_CT_DIR, PLOT_QC_DIR]:
    if not os.path.exists(path):
        os.makedirs(path)

# -------------------------------------------------------------------------
# [Module 2] 数据加载与基因名标准化
# -------------------------------------------------------------------------
print("\n[Module 2] Loading & Normalizing Data...")

# 1. 读取空间转录组数据 (ST)
print(f"    Loading Spatial data from: {DATA_DIR}")
adata_st = sc.read_visium(path=DATA_DIR, count_file='filtered_feature_bc_matrix.h5')

# 2. 读取单细胞数据 (SC)
print(f"    Loading Single-Cell data from: {SC_REF_PATH}")
adata_sc = sc.read_h5ad(SC_REF_PATH)

# [Critical Fix] 基因名标准化 (基于诊断结果)
# 强制转为全大写，确保 'cd3d' 和 'CD3D' 能匹配上
print("    [Fix] Standardizing gene names to UPPER CASE...")
adata_st.var_names = adata_st.var_names.str.upper()
adata_sc.var_names = adata_sc.var_names.str.upper()

# 去重
adata_st.var_names_make_unique()
adata_sc.var_names_make_unique()

# 打印数据概况
print(f"    Spatial Data Shape: {adata_st.shape}")
print(f"    SC Data Shape:      {adata_sc.shape}")

# 检查重合基因数
overlap_genes = np.intersect1d(adata_sc.var_names, adata_st.var_names)
print(f"    [Check] Overlapping training genes: {len(overlap_genes)}")

# -------------------------------------------------------------------------
# [Module 3] 预处理 (Preprocessing)
# -------------------------------------------------------------------------
print("\n[Module 3] Preprocessing...")

# 1. 标准化与对数化 (Normalize & Log1p)
# 这是一个非常标准的单细胞预处理流程
sc.pp.normalize_total(adata_sc, target_sum=1e4)
sc.pp.log1p(adata_sc)
sc.pp.normalize_total(adata_st, target_sum=1e4)
sc.pp.log1p(adata_st)

# 2. 计算 Marker 基因 (寻找训练特征)
# Tangram 需要靠这些差异基因来定位细胞
print("    Calculating marker genes (Rank Genes Groups)...")
# 假设你的单细胞注释列名为 'Subset'，如果不是请修改
sc.tl.rank_genes_groups(adata_sc, groupby="Subset", use_raw=False)

# 提取 Top 100 Marker 基因
markers_df = pd.DataFrame(adata_sc.uns["rank_genes_groups"]["names"]).iloc[0:100, :]
markers = list(np.unique(markers_df.melt().value.values))
print(f"    Identified {len(markers)} marker genes for training.")

# 3. 备份完整数据 (用于后续的全基因组插补)
adata_sc_full = adata_sc.copy()

# 4. 裁剪数据 (Training Prep)
# 只保留 SC 和 ST 中都存在的 Marker 基因用于训练
tg.pp_adatas(adata_sc, adata_st, genes=markers)
clean_gpu()

# -------------------------------------------------------------------------
# [Module 4] 模型训练 (Model Training) - 含智能跳过
# -------------------------------------------------------------------------
print("\n[Module 4] Tangram Model Training...")

map_file_path = os.path.join(OUT_DIR, "ad_map.h5ad")

# [Smart Skip] 检查是否已存在训练好的模型
if os.path.exists(map_file_path):
    print(f"    [INFO] Found existing model: {map_file_path}")
    print("    [INFO] Skipping training to save time. Loading model directly...")
    ad_map = sc.read_h5ad(map_file_path)
else:
    print("    [INFO] No model found. Starting training (Approx. 5-10 mins)...")
    # 核心训练函数
    # mode='cells': 将每个单细胞映射到空间
    # density_prior: 基于 RNA 总量校正密度
    ad_map = tg.map_cells_to_space(
        adata_sc,
        adata_st,
        mode="cells",
        density_prior='rna_count_based',
        num_epochs=1000,
        device=device
    )
    # 保存模型
    ad_map.write_h5ad(map_file_path)
    print("    Training completed and model saved.")

# 生成 QC 图：映射概率直方图
# 评估模型是否确信地将细胞分配到了特定位置
print("    Generating Mapping Confidence Histogram...")
plt.figure(figsize=(8, 6))
X_data = ad_map.X
# 展平数据以绘制直方图
probs_flat = X_data.data if issparse(X_data) else np.asarray(X_data).flatten()
plt.hist(np.log1p(probs_flat), bins=100, color='skyblue', edgecolor='black')
plt.title("Distribution of Mapping Probabilities (log1p)")
plt.xlabel("Log1p(Probability)")
plt.ylabel("Frequency")
plt.savefig(os.path.join(PLOT_QC_DIR, "mapping_probability_hist.png"))
plt.close()

clean_gpu()

# -------------------------------------------------------------------------
# [Module 5] 细胞类型投影 (Cell Type Projection)
# -------------------------------------------------------------------------
print("\n[Module 5] Projecting Cell Types to Space...")

# 1. 将单细胞注释投影到空间坐标
tg.project_cell_annotations(ad_map, adata_st, annotation="Subset")

# 2. 导出预测结果 (CSV)
# 这是一个 spot x cell_type 的矩阵，表示每个 spot 里有多少该类型的细胞
df_predictions = adata_st.obsm['tangram_ct_pred']
csv_path = os.path.join(OUT_DIR, "cell_type_predictions.csv")
df_predictions.to_csv(csv_path)
print(f"    Saved predictions matrix to: {csv_path}")

# 3. 批量绘制空间分布图
cell_types = df_predictions.columns.tolist()
print(f"    Generating spatial plots for {len(cell_types)} cell types...")

for ct in cell_types:
    try:
        # 处理文件名非法字符
        safe_name = ct.replace('/', '_').replace(' ', '_')

        fig, ax = plt.subplots(figsize=(6, 6))
        # 绘图核心函数
        tg.plot_cell_annotation_sc(
            adata_st,
            [ct],
            perc=0.02,
            scale_factor=None
        )
        plt.title(f"Spatial: {ct}")
        plt.savefig(os.path.join(PLOT_CT_DIR, f"{safe_name}.png"), bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"    [Warning] Failed to plot {ct}: {e}")

# -------------------------------------------------------------------------
# [Module 6] 基因插补 (Gene Imputation)
# -------------------------------------------------------------------------
print("\n[Module 6] Projecting Gene Expression (Imputation)...")

# 1. 全基因组插补
# 利用单细胞数据填补空间数据中没测到的基因
ad_ge = tg.project_genes(adata_map=ad_map, adata_sc=adata_sc_full)
ad_ge.write_h5ad(os.path.join(OUT_DIR, "ad_ge.h5ad"))

# 2. 导出部分重点基因 (Markers)
print("    Exporting imputed marker genes to CSV...")
# 确保只导出存在的基因
valid_markers = [g for g in markers if g in ad_ge.var_names]
df_imputed = pd.DataFrame(
    ad_ge[:, valid_markers].X.toarray() if issparse(ad_ge.X) else ad_ge[:, valid_markers].X,
    index=ad_ge.obs_names,
    columns=valid_markers
)
df_imputed.to_csv(os.path.join(OUT_DIR, "imputed_genes_subset.csv"))

# 3. 生成 QC 对比图 (QQ Plot)
# 对比真实空间表达值 vs 预测表达值，评估模型准确性
print(f"    Generating gene prediction QC plots...")

# 随机选 4 个基因进行画图检查
check_genes = list(np.random.choice(valid_markers, min(4, len(valid_markers)), replace=False))

if len(check_genes) > 0:
    try:
        tg.plot_genes_sc(
            genes=check_genes,
            adata_measured=adata_st,
            adata_predicted=ad_ge,
            perc=0.02
        )
        plt.savefig(os.path.join(PLOT_QC_DIR, "gene_prediction_comparison.png"), bbox_inches='tight')
        plt.close()
        print("    [Success] QC plot saved.")
    except Exception as e:
        print(f"    [Warning] Could not generate QC plot: {e}")
else:
    print("    [Warning] No overlapping marker genes found for QC plot.")

clean_gpu()
print("=" * 60)
print("Tangram Analysis Pipeline Completed.")
print(f"Results Directory: {OUT_DIR}")
print("=" * 60)