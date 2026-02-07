# ===========================================================================
# Project: Spatial Transcriptomics - Advanced Cell Segmentation and Aggregation
# File Name: 29_cell_segmentation_and_binning_fixed.py
# Author: Kuroneko
# Date: 2026-02-04
# Current Path: F:\ST\code
#
# ---------------------------------------------------------------------------
# 1. 脚本功能描述 (Script Description)
# ---------------------------------------------------------------------------
# 本脚本旨在通过深度学习算法实现跨平台的细胞识别与数据重构：
# - 模块 A (Visium HD): 针对 2um 细分网格数据，利用 StarDist 识别 H&E 影像中的细胞核。
#   通过空间坐标映射，将落在核区域内的网格信号聚合，生成完善的单细胞级表达矩阵。
# - 模块 B (Xenium): 针对原位单细胞影像，利用 Cellpose 3.0 优化细胞边界分割。
# - 模块 C (Visualization): 生成分割效果对比图，用于质量评估。
#
# ---------------------------------------------------------------------------
# 2. 原始数据来源 (Data Acquisition and Origin)
# ---------------------------------------------------------------------------
# - Visium HD (Mouse Small Intestine):
#   数据由 10x Genomics 官方 SpaceRanger HD 流程产生。测序得到的转录本被映射到
#   2um 的微小正方形网格 (Bins) 中。为了获得具有生物学意义的单细胞数据，必须利用
#   H&E 染色影像进行细胞核检测并执行空间信号聚合。
# - Xenium (Human Lymph Node):
#   数据由 10x Xenium 平台通过原位杂交和循环荧光成像产生。脚本使用的影像文件
#   (morphology_focus) 是由仪器机载软件从多层焦深图像中筛选出的最清晰图像融合而成。
#
# ---------------------------------------------------------------------------
# 3. 输入文件清单 (Input Files Mapping)
# ---------------------------------------------------------------------------
# - 组织影像 (Visium HD): F:\ST\code\data\seg\stardist\Visium_HD_Mouse_Small_Intestine_tissue_image.btf
# - 表达矩阵 (Visium HD): F:\ST\code\data\seg\stardist\Visium_HD_Mouse_Small_Intestine_binned_outputs\binned_outputs\square_002um\filtered_feature_bc_matrix.h5
# - 坐标文件 (Visium HD): F:\ST\code\data\seg\stardist\Visium_HD_Mouse_Small_Intestine_binned_outputs\binned_outputs\square_002um\spatial\tissue_positions.parquet
# - 形态影像 (Xenium): F:\ST\code\data\seg\Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_outs\morphology_focus\morphology_focus_0003.ome.tif
#
# ---------------------------------------------------------------------------
# 4. 输出文件清单 (Output Files Mapping)
# ---------------------------------------------------------------------------
# - 细胞级数据: F:\ST\code\results\seg\adata_cell_level.h5ad
# - 分割校验图: F:\ST\code\results\seg\segmentation_preview.png
# ===========================================================================

import os
import pandas as pd
import numpy as np
import scanpy as sc
import anndata
import geopandas as gpd
from tifffile import imread
from csbdeep.utils import normalize
from stardist.models import StarDist2D
from shapely.geometry import Polygon, Point
from scipy import sparse
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# [Module 1] 环境与路径初始化
# ---------------------------------------------------------------------------
DATA_ROOT = r"F:\ST\code\data\seg"
OUT_DIR = r"F:\ST\code\results\seg"
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

# ---------------------------------------------------------------------------
# [Module 2] StarDist 细胞核分割 (Visium HD 流程)
# 算法理由: StarDist 利用深度学习模型预测星形凸多边形，能有效识别 H&E 影像中
# 紧密堆叠的细胞核，是聚合 2um 信号的核心前提。
# ---------------------------------------------------------------------------
print("Starting StarDist Segmentation...")

# 读取官方 Visium HD 高清影像
img_path = os.path.join(DATA_ROOT, "stardist", "Visium_HD_Mouse_Small_Intestine_tissue_image.btf")
img = imread(img_path)

# 图像归一化 (5-95 百分位)
img_norm = normalize(img, 5, 95)

# 加载预训练模型 (需管理员权限或开启开发人员模式以创建模型链接)
model = StarDist2D.from_pretrained('2D_versatile_he')

# [FIX] 增加 min_overlap 参数，防止大图分块处理时切断边缘细胞核
# block_size 可根据显存大小调整 (4096 适合大显存，2048 适合中等显存)
labels, polys = model.predict_instances_big(img_norm, axes='YXC', block_size=4096, min_overlap=128)

# 将检测结果转换为地理坐标系 (Polygon)
geometries = [Polygon([(y, x) for x, y in zip(p[0], p[1])]) for p in polys['coord']]
gdf_nuclei = gpd.GeoDataFrame(geometry=geometries)
gdf_nuclei['id'] = [f"Nuclei_{i}" for i in range(len(gdf_nuclei))]

# 可视化分割预览图 (前500个细胞，避免绘图过慢)
plt.figure(figsize=(10, 10))
plt.imshow(img_norm, cmap='gray')
gdf_nuclei.iloc[:500].boundary.plot(ax=plt.gca(), color='red', linewidth=0.5)
plt.title("Segmentation Preview (Red Outlines)")
plt.savefig(os.path.join(OUT_DIR, "segmentation_preview.png"))
plt.close()

# ---------------------------------------------------------------------------
# [Module 3] 空间聚合: 重构单细胞级数据
# 算法理由: 将 2um 的网格点通过空间包含关系 (Point-in-Polygon) 映射到细胞核
# 多边形中。将落在同一个核内的所有网格表达量相加，获得真实细胞特征。
# ---------------------------------------------------------------------------
print("Aggregating 2um Bin signals...")

# 定位 Visium HD 数据子目录
hd_base = os.path.join(DATA_ROOT, "stardist", "Visium_HD_Mouse_Small_Intestine_binned_outputs", "binned_outputs", "square_002um")

# 1. 加载矩阵 (Expression Matrix)
adata_bin = sc.read_10x_h5(os.path.join(hd_base, "filtered_feature_bc_matrix.h5"))
adata_bin.var_names_make_unique()

# 2. 加载坐标 (Coordinates)
df_pos = pd.read_parquet(os.path.join(hd_base, "spatial", "tissue_positions.parquet"))

# 数据对齐 (Data Alignment)
if 'barcode' in df_pos.columns:
    df_pos = df_pos.set_index('barcode')

# 仅保留在 filtered 矩阵中存在的 Barcodes
valid_barcodes = adata_bin.obs_names.intersection(df_pos.index)
print(f"Aligning data: {len(valid_barcodes)} bins matched between Coordinates and Matrix.")

adata_bin = adata_bin[valid_barcodes, :].copy()
df_pos = df_pos.loc[valid_barcodes]

# 3. 构建空间点对象
gdf_points = gpd.GeoDataFrame(
    df_pos,
    geometry=[Point(xy) for xy in zip(df_pos['pxl_col_in_fullres'], df_pos['pxl_row_in_fullres'])]
)

# 4. 空间连接 (Spatial Join)
# 这一步确定了 Grid -> Nuclei 的映射关系
res_join = gpd.sjoin(gdf_points, gdf_nuclei, how='inner', predicate='within')

if res_join.empty:
    raise ValueError("No bins fell inside the nuclei polygons. Check coordinate scaling or registration.")

# 提取映射后的数据
filtered_obs = adata_bin[res_join.index, :].copy()
filtered_obs.obs['target_id'] = res_join['id']

# [CRITICAL FIX] 内存优化聚合算法
# 放弃 pd.get_dummies，改用 scipy.sparse 直接构建聚合矩阵
# -----------------------------------------------------------------------
print("Performing sparse matrix aggregation...")

# A. 准备映射索引
# 将字符串形式的 'target_id' (细胞ID) 映射为整数索引 (0, 1, 2...)
groups = filtered_obs.obs['target_id']
unique_cells = groups.unique()
cell_to_idx = {cid: i for i, cid in enumerate(unique_cells)}
cell_indices = groups.map(cell_to_idx).values

n_cells = len(unique_cells)
n_bins = filtered_obs.n_obs

# B. 构建聚合矩阵 M (Shape: Cells x Bins)
# M[i, j] = 1 表示第 j 个 Bin 属于第 i 个 Cell
# Row indices = cell_indices (每个 Bin 对应的 Cell 编号)
# Col indices = np.arange(n_bins) (Bin 的序号)
from scipy import sparse
row_idx = cell_indices
col_idx = np.arange(n_bins)
data = np.ones(n_bins, dtype=np.float32)

# 直接构建 CSR 稀疏矩阵 (极度节省内存)
M = sparse.coo_matrix((data, (row_idx, col_idx)), shape=(n_cells, n_bins)).tocsr()

# C. 矩阵乘法聚合
# (Cells x Bins) * (Bins x Genes) = (Cells x Genes)
aggregated_X = M.dot(filtered_obs.X)

# -----------------------------------------------------------------------

# 创建完善的 AnnData 对象
adata_cell = anndata.AnnData(
    X=aggregated_X,
    obs=pd.DataFrame(index=unique_cells, data={'cell_id': unique_cells}),
    var=adata_bin.var
)

print(f"Aggregation complete. Generated {adata_cell.n_obs} single cells.")

# 保存聚合后的单细胞级数据
adata_cell.write_h5ad(os.path.join(OUT_DIR, "adata_cell_level.h5ad"))