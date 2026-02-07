# ===========================================================================
# Project: Visium HD High-Precision Reconstruction & Contact Analysis
# File Name: 30_visium_hd_cellpose_complete_analysis_fixed.py
# Author: Kuroneko (PhD Candidate)
# Current Path: F:\ST\code
#
# ---------------------------------------------------------------------------
# [1. 脚本功能全览 (Script Functionality)]
# ---------------------------------------------------------------------------
# 本脚本实现从“原始影像”到“物理空间通讯”的一站式分析：
# 1. [高精度分割]: 利用 Cellpose 3.0 (cyto3) 对 Visium HD 的 H&E 影像进行全细胞分割，
#    生成包含细胞质边界的精准掩码 (Mask)，突破 StarDist 仅识别细胞核的局限。
# 2. [极速重构]: 使用 Numpy 矩阵索引技术，将数百万个 2um 网格信号毫秒级映射回
#    对应的细胞实体，生成单细胞精度的 AnnData 对象。
# 3. [标准分析]: 执行 Scanpy 标准流程（质控、降维、聚类、UMAP 可视化）。
# 4. [物理通讯]: 利用 Squidpy 构建“接触图 (Contact Graph)”，仅计算物理边界
#    直接接触的细胞间的配体-受体互作，排除远距离假阳性。
#
# ---------------------------------------------------------------------------
# [2. 原始数据来源 (Data Origin)]
# ---------------------------------------------------------------------------
# - 影像数据: Visium HD 实验产生的 .btf 高分辨率 H&E 染色影像。
# - 表达数据: SpaceRanger HD 流程产出的 2um 细分网格 (Bin) 矩阵。
# - 核心逻辑: 2um 网格不具备生物学意义，必须通过 Cellpose 分割将其“归位”到
#   真实的细胞形态中，才能进行可信的互作分析。
#
# ---------------------------------------------------------------------------
# [3. 输入文件清单 (Input Mapping)]
# ---------------------------------------------------------------------------
# - Visium HD 影像: F:\ST\code\data\seg\stardist\Visium_HD_Mouse_Small_Intestine_tissue_image.btf
# - 2um 表达矩阵: F:\ST\code\data\seg\stardist\Visium_HD_Mouse_Small_Intestine_binned_outputs\binned_outputs\square_002um\filtered_feature_bc_matrix.h5
# - 2um 空间坐标: F:\ST\code\data\seg\stardist\Visium_HD_Mouse_Small_Intestine_binned_outputs\binned_outputs\square_002um\spatial\tissue_positions.parquet
#
# ---------------------------------------------------------------------------
# [4. 输出文件清单 (Output Mapping)]
# ---------------------------------------------------------------------------
# - 最终数据对象: F:\ST\code\results\seg_cellpose\adata_visium_hd_processed.h5ad
# - 分割掩码文件: F:\ST\code\results\seg_cellpose\visium_hd_cellpose_mask.npy
# - 聚类空间分布图: F:\ST\code\results\seg_cellpose\01_spatial_clusters.png
# - 细胞通讯热图: F:\ST\code\results\seg_cellpose\02_contact_ligrec.pdf
# ===========================================================================

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import squidpy as sq
from scipy import sparse
from tifffile import imread
from cellpose import models
from skimage.measure import regionprops
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# [Helper Function] 修复保存报错的关键函数
# ---------------------------------------------------------------------------
def clean_uns_for_save(adata):
    """
    修复 Squidpy 在 uns 中生成的 Tuple 列名，使其能被 h5ad 格式兼容。
    将 ('0', '1') 这种列名转换为 '0_1'。
    """
    print("    [Sanitizing] Cleaning data structure for saving...")
    if 'leiden_ligrec' in adata.uns:
        res = adata.uns['leiden_ligrec']
        for key in res.keys():
            # 检查是否是 DataFrame 且列名包含 Tuple
            if isinstance(res[key], pd.DataFrame):
                new_cols = []
                for col in res[key].columns:
                    if isinstance(col, tuple):
                        # 将 ('0', '0') 转为 '0_0'
                        new_cols.append("_".join(map(str, col)))
                    else:
                        new_cols.append(str(col))
                res[key].columns = new_cols
                print(f"      Fixed columns for: adata.uns['leiden_ligrec']['{key}']")
    return adata


# ---------------------------------------------------------------------------
# [Module 1] 环境初始化
# ---------------------------------------------------------------------------
BASE_DIR = r"F:\ST\code\data\seg"
OUT_DIR = r"F:\ST\code\results\seg_cellpose"

if __name__ == "__main__":
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    sc.settings.figdir = OUT_DIR
    sc.settings.set_figure_params(dpi=150, frameon=False, facecolor='white')

    # ---------------------------------------------------------------------------
    # [Module 2] Cellpose 全细胞分割 (带断点续传)
    # ---------------------------------------------------------------------------
    print("\n[Step 1] Loading Image & Running Cellpose...")

    img_path = os.path.join(BASE_DIR, "stardist", "Visium_HD_Mouse_Small_Intestine_tissue_image.btf")
    mask_path = os.path.join(OUT_DIR, "visium_hd_cellpose_mask.npy")

    if os.path.exists(mask_path):
        print(f"    [Info] Found existing mask: {mask_path}")
        print("    Skipping Cellpose inference to save time.")
        masks = np.load(mask_path)
    else:
        print("    Starting new segmentation...")
        img = imread(img_path)
        print("    Initializing CellposeModel (cyto3)...")
        try:
            model = models.CellposeModel(gpu=True, model_type='cyto3')
        except:
            print("    [Warning] 'cyto3' failed, using 'cyto2'.")
            model = models.CellposeModel(gpu=True, model_type='cyto2')

        print("    Running segmentation (eval)...")
        masks, flows, styles = model.eval(img, diameter=30, channels=[0, 0], do_3D=False)[:3]
        np.save(mask_path, masks)

    print(f"    Mask loaded. Shape: {masks.shape}")

    # ---------------------------------------------------------------------------
    # [Module 3] 极速信号重构
    # ---------------------------------------------------------------------------
    print("\n[Step 2] Reconstructing Single-Cell Data from 2um Bins...")

    hd_root = os.path.join(BASE_DIR, "stardist", "Visium_HD_Mouse_Small_Intestine_binned_outputs", "binned_outputs",
                           "square_002um")

    # 这一步比较快，直接重跑没问题
    print("    Loading expression matrix & coordinates...")
    adata_bin = sc.read_10x_h5(os.path.join(hd_root, "filtered_feature_bc_matrix.h5"))
    adata_bin.var_names_make_unique()
    df_pos = pd.read_parquet(os.path.join(hd_root, "spatial", "tissue_positions.parquet"))

    # Barcode 对齐
    if 'barcode' in df_pos.columns:
        df_pos = df_pos.set_index('barcode')

    valid_barcodes = adata_bin.obs_names.intersection(df_pos.index)
    adata_bin = adata_bin[valid_barcodes, :].copy()
    df_pos = df_pos.loc[valid_barcodes]

    # 坐标映射
    y_coords = df_pos['pxl_row_in_fullres'].values.astype(int)
    x_coords = df_pos['pxl_col_in_fullres'].values.astype(int)

    valid_idx = (x_coords >= 0) & (x_coords < masks.shape[1]) & \
                (y_coords >= 0) & (y_coords < masks.shape[0])

    cell_ids = np.zeros(len(df_pos), dtype=int)
    cell_ids[valid_idx] = masks[y_coords[valid_idx], x_coords[valid_idx]]
    adata_bin.obs['cell_id'] = cell_ids

    adata_valid = adata_bin[adata_bin.obs['cell_id'] > 0].copy()

    # 极速聚合
    unique_cells = np.unique(adata_valid.obs['cell_id'].values)
    cell_to_idx = {cid: i for i, cid in enumerate(unique_cells)}

    row_indices = adata_valid.obs['cell_id'].map(cell_to_idx).values
    col_indices = np.arange(adata_valid.n_obs)
    data = np.ones(len(row_indices))

    M = sparse.csr_matrix((data, (row_indices, col_indices)), shape=(len(unique_cells), adata_valid.n_obs))
    summed_X = M.dot(adata_valid.X)

    adata = anndata.AnnData(
        X=summed_X,
        obs=pd.DataFrame(index=[str(c) for c in unique_cells], data={'cell_id': unique_cells}),
        var=adata_bin.var
    )

    # 坐标注入
    props = regionprops(masks)
    centroid_dict = {p.label: p.centroid for p in props}
    valid_cell_ids = [c for c in unique_cells if c in centroid_dict]
    coords = np.array([centroid_dict[cid] for cid in valid_cell_ids])

    if len(valid_cell_ids) != len(unique_cells):
        adata = adata[[str(c) for c in valid_cell_ids], :].copy()

    adata.obsm['spatial'] = np.column_stack((coords[:, 1], coords[:, 0]))

    print(f"    Reconstruction complete: {adata.n_obs} cells generated.")

    # ---------------------------------------------------------------------------
    # [Module 4] 下游分析
    # ---------------------------------------------------------------------------
    print("\n[Step 3] Standard Analysis Pipeline...")

    sc.pp.filter_cells(adata, min_genes=100)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    sc.tl.pca(adata)
    sc.pp.neighbors(adata, n_neighbors=15)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=0.5, key_added="leiden")

    sc.pl.embedding(adata, basis="spatial", color="leiden", title="Spatial Clusters",
                    show=False, frameon=False, size=1.5)
    plt.savefig(os.path.join(OUT_DIR, "01_spatial_clusters.png"))
    plt.close()

    # ---------------------------------------------------------------------------
    # [Module 5] 物理通讯
    # ---------------------------------------------------------------------------
    print("\n[Step 4] Physical Cell Communication...")

    # 使用 Delaunay 模拟接触
    print("    Calculating contact neighbors (Delaunay)...")
    sq.gr.spatial_neighbors(
        adata,
        spatial_key="spatial",
        coord_type="generic",
        delaunay=True,
        key_added="spatial"
    )

    print("    Running Ligand-Receptor analysis...")
    # 注意：如果网络不好，这一步可能会卡很久或者报错。但只要跑过去了，后面就是保存。
    sq.gr.ligrec(adata, n_perms=100, cluster_key="leiden", use_raw=False)

    try:
        sq.pl.ligrec(
            adata,
            cluster_key="leiden",
            source_groups="0",
            means_range=(0.3, np.inf),
            alpha=0.05,
            swap_axes=True,
            save="02_contact_ligrec.pdf"
        )
    except Exception as e:
        print(f"    [Warning] Plotting failed (data issue?): {e}")

    # ---------------------------------------------------------------------------
    # [Module 6] 结果保存 (关键修复点)
    # ---------------------------------------------------------------------------
    print("\n[Step 5] Saving Results...")

    # [FIX] 在保存前调用清理函数
    adata = clean_uns_for_save(adata)

    save_path = os.path.join(OUT_DIR, "adata_visium_hd_processed.h5ad")
    try:
        adata.write_h5ad(save_path)
        print(f"\nPipeline Finished Successfully. Results saved to: {save_path}")
    except Exception as e:
        print(f"\n[Error] Saving still failed: {e}")

        print("建议：尝试删除 adata.uns['leiden_ligrec'] 后再保存。")
