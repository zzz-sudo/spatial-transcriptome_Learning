# -*- coding: utf-8 -*-
"""
---------------------------------------------------------------------------
文件名称: st_full_process_from_csv.py
作    者: Kuroneko


构建单细胞参考集
文件名称:
    st_build_reference.py
功能:
    读取原始 CSV -> 组装 AnnData -> 预训练 (PCA/Neighbors) -> 保存
输出:
    F:/ST/code/02/adata_processed.h5ad (这是给下一步用的标准参考集)

也就是说空间转录组的注释需要借助单细胞对吧，但是这个注释其实就是说，
一个spot主要特征的注释对吧？其次我们先整合两个空转文件在注释，这个代码的核心就是去批次多样本整合+注释

"""

import scanpy as sc
import pandas as pd
import os
import gc

# 1. 设置路径
base_dir = r"F:\ST\code"
input_dir = os.path.join(base_dir, "02")
# 我们把做好的参考集直接存回 02 文件夹，假装它是下载好的
output_path = os.path.join(input_dir, "adata_processed.h5ad")

print("正在启动单细胞参考集构建程序...")

# 2. 读取表达矩阵 (2.3GB)
print("--> 读取表达矩阵 CSV (请耐心等待)...")
csv_path = os.path.join(input_dir, "GSE115746_cells_exon_counts.csv")
df_counts = pd.read_csv(csv_path, index_col=0)

# 转置 (确保 行=细胞)
if df_counts.shape[0] > df_counts.shape[1]:
    print("检测到基因在行，执行转置...")
    df_counts = df_counts.T

adata = sc.AnnData(df_counts)
del df_counts
gc.collect()

# 3. 读取元数据
print("--> 读取元数据并对齐...")
meta_path = os.path.join(input_dir, "GSE115746_complete_metadata_28706-cells.csv")
df_meta = pd.read_csv(meta_path, index_col=0)

# 取交集
common_cells = adata.obs_names.intersection(df_meta.index)
adata = adata[common_cells, :]
adata.obs = df_meta.loc[common_cells]

# 4. 预训练参考集 (关键步骤!)
# 这一步是为了让这个 h5ad "学会" 数据的结构，下一步 Ingest 直接用
print("--> 预训练参考集 (归一化/PCA/Neighbors)...")
adata.var_names_make_unique()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)
sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata) # 顺便算个 UMAP，虽然不是必须，但画图有用

# 5. 保存成品
print(f"--> 保存最终参考集至: {output_path}")
adata.write(output_path)

print("完成！")
