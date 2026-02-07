# -*- coding: utf-8 -*-
"""
===========================================================================
File Name: st_mouse_tangram_official_fix.py
Author: Kuroneko
Date: 2026-02-02
Version: V25.0

Description:
    [Mouse Brain Spatial Analysis - Official Tangram Workflow]

    This script strictly follows the official Tangram tutorial logic while
    fixing the critical 'RuntimeError' on Windows.

    CRITICAL FIXES:
    1. Windows Safety: All execution logic is wrapped in 'if __name__ == "__main__":'.
    2. Stability: Squidpy functions use n_jobs=1 to avoid multiprocessing crashes.
    3. Official Logic: Uses 'tg.pp_adatas' instead of manual parameter injection.

    Included Modules:
    1. Data Loading: Load single-cell and spatial datasets.
    2. Data Preprocessing: Calculate markers -> tg.pp_adatas().
    3. Tangram Mapping: Train model and project cell types.
    4. Model Validation: Assess mapping accuracy (AUC).
    5. Spatial Graph: Construct spatial neighbor graph.
    6. Neighborhood Enrichment: Analyze cell-type co-localization (Safe Mode).
    7. Cell Communication: Infer Ligand-Receptor interactions (Safe Mode).
    8. Spatial Variability: Identify spatially variable genes (Moran's I).
    9. Export: Save final results and statistical tables.

Input Files:
    - F:/ST/code/03/adata_sc.h5ad
    - F:/ST/code/03/adata_st.h5ad

Output Files:
    - F:/ST/code/results/mouse_final_result.h5ad
    - F:/ST/code/results/spatial_variable_genes.csv
    - F:/ST/code/results/figures_mouse/
===========================================================================
"""

import os
import sys
import warnings
import scanpy as sc
import squidpy as sq
import tangram as tg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

# =========================================================================
# Main Execution Block (Required for Windows)
# =========================================================================
if __name__ == "__main__":

    # 1. System Initialization (系统初始化)
    print("[Module 0] Initializing system environment...")

    if os.path.basename(__file__) == "tangram.py":
        print("[Error] Please rename this script. It cannot be named tangram.py")
        sys.exit(1)

    # 屏蔽警告
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.filterwarnings("ignore", message="The legacy Dask DataFrame implementation")

    BASE_DIR = r"F:\ST\code\03"
    RESULT_DIR = r"F:\ST\code\results"
    FIGURE_DIR = os.path.join(RESULT_DIR, "figures_mouse")

    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    if not os.path.exists(FIGURE_DIR):
        os.makedirs(FIGURE_DIR)

    sc.settings.verbosity = 3
    sc.settings.figdir = FIGURE_DIR
    sc.set_figure_params(dpi=150, facecolor="white", vector_friendly=True)

    if torch.cuda.is_available():
        device = "cuda:0"
        print(f"    [Status] GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("    [Status] GPU not found. Using CPU mode.")

    # 2. Data Loading (数据加载)
    print("\n[Module 1] Data Loading...")
    sc_path = os.path.join(BASE_DIR, "adata_sc.h5ad")
    st_path = os.path.join(BASE_DIR, "adata_st.h5ad")

    if not (os.path.exists(sc_path) and os.path.exists(st_path)):
        print("[Error] Input files not found.")
        sys.exit(1)

    adata_sc = sc.read_h5ad(sc_path)
    adata_st = sc.read_h5ad(st_path)

    # 确保基因名唯一
    adata_sc.var_names_make_unique()
    adata_st.var_names_make_unique()

    # 简单的归一化 (Standard Scanpy Preprocessing)
    print("    Normalizing single-cell data...")
    sc.pp.normalize_total(adata_sc)

    # 识别细胞类型列
    potential_keys = ['cell_subclass', 'cell_cluster', 'CellType', 'leiden']
    sc_label_key = next((k for k in potential_keys if k in adata_sc.obs.columns), None)
    print(f"    Using cell type column: {sc_label_key}")

    # 3. Data Preprocessing (数据预处理 - 官网流程)
    print("\n[Module 2] Data Preprocessing (Official tg.pp_adatas)...")

    # 第一步：计算 Marker Genes (模拟官网的 markers.csv)
    print("    Generating marker genes dynamically...")
    sc.tl.rank_genes_groups(adata_sc, groupby=sc_label_key, use_raw=False)
    markers_df = pd.DataFrame(adata_sc.uns["rank_genes_groups"]["names"]).iloc[:100, :]
    markers = list(set(markers_df.melt().value.values))
    print(f"    Identified {len(markers)} marker genes.")

    # 第二步：调用官方函数 tg.pp_adatas
    # 这会自动计算 training_genes, density_prior 并存入 uns/obs
    print("    Running tg.pp_adatas()...")
    tg.pp_adatas(adata_sc, adata_st, genes=markers)
    print("    tg.pp_adatas completed successfully.")

    # 4. Tangram Mapping (模型训练与映射)
    print("\n[Module 3] Tangram Mapping...")

    # 官方调用方式，不再需要手动切片，tg会自动处理
    ad_map = tg.map_cells_to_space(
        adata_sc,
        adata_st,
        mode="cells",
        density_prior='rna_count_based',
        num_epochs=500,
        device=device
    )

    print("    Model training finished.")

    # 投影细胞类型
    print("    Projecting cell types...")
    tg.project_cell_annotations(ad_map, adata_st, annotation=sc_label_key)

    # 结果固化
    df_probs = adata_st.obsm["tangram_ct_pred"]
    adata_st.obs["tangram_max_celltype"] = df_probs.idxmax(axis=1)

    # 绘图
    sc.pl.spatial(adata_st, color="tangram_max_celltype", title="Tangram Prediction",
                  frameon=False, show=False, save="_Tangram_Official_Pred.png")
    print("    Prediction plot saved.")

    # 5. Model Validation (模型验证)
    print("\n[Module 4] Model Validation (AUC)...")

    # 投影基因表达
    ad_ge = tg.project_genes(adata_map=ad_map, adata_sc=adata_sc)

    # 计算 AUC (Tangram 会自动处理 overlap genes)
    try:
        df_all_genes = tg.compare_spatial_geneexp(ad_ge, adata_st, adata_sc)
        tg.plot_auc(df_all_genes)
        plt.savefig(os.path.join(FIGURE_DIR, "Tangram_Validation_AUC.png"))
        plt.close()
        print("    Validation plot saved.")
    except Exception as e:
        print(f"    [Warning] Validation skipped: {e}")

    # 6. Spatial Graph (构建空间图)
    print("\n[Module 5] Constructing Spatial Graph...")
    sq.gr.spatial_neighbors(adata_st, coord_type="generic", n_neighs=6)

    # 7. Neighborhood Enrichment (邻域富集 - 安全模式)
    print("\n[Module 6] Neighborhood Enrichment...")

    # 安全检查：是否只有一种细胞类型？
    n_clusters = len(adata_st.obs["tangram_max_celltype"].unique())
    if n_clusters < 2:
        print(f"    [Warning] Only {n_clusters} cell type found. Skipping enrichment analysis.")
    else:
        # CRITICAL: n_jobs=1 prevents Windows crash
        sq.gr.nhood_enrichment(adata_st, cluster_key="tangram_max_celltype", n_jobs=1)

        sq.pl.nhood_enrichment(adata_st, cluster_key="tangram_max_celltype",
                               method="ward", title="Neighborhood Enrichment",
                               show=False, save="_Neighborhood_Enrichment.png")
        print("    Enrichment analysis done.")

    # 8. Cell Communication (细胞通讯 - 安全模式)
    print("\n[Module 7] Cell Communication (Ligand-Receptor)...")
    if n_clusters < 2:
        print("    [Skip] Not enough cell types for interaction analysis.")
    else:
        try:
            # CRITICAL: n_jobs=1
            sq.gr.ligand_receptor(
                adata_st,
                cluster_key="tangram_max_celltype",
                interaction_params={"test_prop": 0.2},
                n_jobs=1
            )

            # Plot
            top_cells = adata_st.obs["tangram_max_celltype"].value_counts().index[:2].tolist()
            if len(top_cells) >= 2:
                s, t = top_cells[0], top_cells[1]
                sq.pl.ligand_receptor(
                    adata_st, cluster_key="tangram_max_celltype",
                    source_groups=s, target_groups=t,
                    title=f"Interaction: {s}-{t}",
                    show=False, save="_LR_Interaction.png"
                )
                print("    Communication analysis done.")
        except Exception as e:
            print(f"    [Warning] Communication skipped: {e}")

    # 9. Spatial Variability (空间高变基因)
    print("\n[Module 8] Spatial Variability (Moran's I)...")
    if "highly_variable" not in adata_st.var.columns:
        sc.pp.highly_variable_genes(adata_st, n_top_genes=2000)

    genes_test = adata_st.var_names[adata_st.var.highly_variable]

    # CRITICAL: n_jobs=1
    sq.gr.spatial_autocorr(adata_st, mode="moran", genes=genes_test, n_perms=100, n_jobs=1)

    if "moranI" in adata_st.uns:
        moran_df = adata_st.uns["moranI"]
        csv_path = os.path.join(RESULT_DIR, "spatial_variable_genes.csv")
        moran_df.to_csv(csv_path)
        print(f"    Spatial variable genes saved to: {csv_path}")

    # 10. Export (结果导出)
    print("\n[Module 9] Exporting Results...")
    save_path = os.path.join(RESULT_DIR, "mouse_final_result_official.h5ad")
    adata_st.write(save_path, compression="gzip")

    print("=" * 60)
    print(f"Pipeline Completed Successfully.")
    print("=" * 60)