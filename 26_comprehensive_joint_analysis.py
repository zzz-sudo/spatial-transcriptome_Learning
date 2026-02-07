# ===========================================================================
# File Name: 26_comprehensive_joint_analysis.py
# Author: Kuroneko (Maize Nutrients Research Team)
# Date: 2026-02-04
# Version: V1.3 (Ultra-Detailed Integrated Pipeline)
#
# Description:
#     [空间转录组多模态数据深度集成与功能景观图谱重建]
#
#     本脚本是整个分析流程的"终章"，旨在将所有碎片化的分析结果串联成一个完整的生物学故事。
#     它通过将细胞比例（丰度）、物理解剖域、以及功能生态位在同一坐标系下进行多维重叠，
#     揭示组织（如玉米节部）内各区域的细胞构成规律与分子执行功能。
#
# ===========================================================================
# 【输入文件清单 (Input Files - Detailed)】
# ---------------------------------------------------------------------------
# 1. results/deconvolution/c2l_results.h5ad:
#    - 关键来源: 22_run_deconvolution_c2l.py
#    - 数据字段: adata.obsm['cell_abundance']
#    - 科学意义: 提供了每个 Spot 中各细胞类型（如韧皮部、伴胞、转运蛋白细胞）的精确估计丰度。
#
# 2. results/spagcn/spagcn_domains.csv:
#    - 关键来源: 24_a_spatial_domains_spagcn.py
#    - 数据字段: 'pred' 或 'spagcn_domain' 列
#    - 科学意义: 定义了物理空间的解剖边界，标识哪些区域属于维管束，哪些属于皮层。
#
# 3. results/niche_analysis/niche_analysis_results.h5ad:
#    - 关键来源: 25_run_niche_analysis_nmf.py
#    - 数据字段: adata.obs['niche_labels']
#    - 科学意义: 揭示了功能上协同工作的"细胞社会"（生态位），超越了单一解剖结构的限制。
#
# 4. results/spagcn/all_domains_svg_summary.csv:
#    - 关键来源: 24_a_spatial_domains_spagcn.py
#    - 科学意义: 存储了各空间域的标志基因，是进行 GO/KEGG 功能注释的核心原始列表。
#
# ===========================================================================
# 【输出文件清单 (Output Files - Detailed)】
# ---------------------------------------------------------------------------
# 存储根目录: F:/ST/code/results/joint_analysis/
#
# 1. 01_Domain_Cell_Composition.pdf (解剖域细胞构成图):
#    - 展示每个解剖分区（Domain）的平均细胞占比，用于确认维管束的细胞学特征。
#
# 2. 02_Niche_Domain_Heatmap.pdf (生态位-解剖域相关性热图):
#    - 展示功能性生态位（Niche）与物理解剖区域（Domain）的对应概率。
#
# 3. 03_Annotation_Domain_X.png (区域功能气泡图):
#    - 详细展示每个空间域富集到的代谢通路，解释该区域为何种生理功能（如：同化物卸载）。
#
# 4. 04_Nutrient_Transporter_Specialized_Heatmap.pdf (养分转运基因图谱):
#    - 针对糖（SUT/SWEET）、水（PIP）等关键转运蛋白的定向表达热图。
#
# 5. detailed_functional_annotation.csv: 完整的富集分析统计表，包含 p-value 和富集倍数。
# ===========================================================================

import os
import pandas as pd
import numpy as np
import scanpy as sc
import gseapy as gp
import matplotlib.pyplot as plt
import seaborn as sns

# ===========================================================================
# [Module 1] 数据加载与对齐
# ===========================================================================
print("[Module 1] Loading data...")
BASE_DIR = r"F:/ST/code/"
RES_DIR = os.path.join(BASE_DIR, "results")
OUT_DIR = os.path.join(RES_DIR, "joint_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

adata = sc.read_h5ad(os.path.join(RES_DIR, "deconvolution/c2l_results.h5ad"))
dom_df = pd.read_csv(os.path.join(RES_DIR, "spagcn/spagcn_domains.csv"), index_col=0)
adata.obs['domain'] = dom_df['pred'].reindex(adata.obs_names).astype('category')
n_adata = sc.read_h5ad(os.path.join(RES_DIR, "niche_analysis/niche_analysis_results.h5ad"))
adata.obs['niche'] = n_adata.obs['niche_labels'].reindex(adata.obs_names).astype('category')

# ===========================================================================
# [Module 2] 细胞构成柱状图
# ===========================================================================
print("[Module 2] Plotting composition...")
comp = adata.obsm['cell_abundance'].copy()
comp['domain'] = adata.obs['domain']
comp.groupby('domain', observed=True).mean().plot(kind='bar', stacked=True, colormap='tab20', figsize=(10, 6))
plt.savefig(os.path.join(OUT_DIR, "01_Composition.pdf"), bbox_inches='tight')
plt.close()

# ===========================================================================
# [Module 3] 生态位热图
# ===========================================================================
print("[Module 3] Plotting niche heatmap...")
cross = pd.crosstab(adata.obs['domain'], adata.obs['niche'], normalize='index')
sns.heatmap(cross, annot=True, cmap="YlOrRd")
plt.savefig(os.path.join(OUT_DIR, "02_Correlation.pdf"))
plt.close()

# ===========================================================================
# [Module 4] 深度功能注释 (解决白图问题：手动绘图)
# ===========================================================================
# -------------------------------------------------------------------------
# [Module 4] 深度功能注释 (解决图例重叠与 PDF 截断)
# -------------------------------------------------------------------------
print("\n[Module 4] Performing Local Enrichment (Layout Optimized)...")

import textwrap  # 用于处理超长路径名

gmt_files = [
    os.path.join(BASE_DIR, "data/human/GO_Biological_Process_2023.gmt"),
    os.path.join(BASE_DIR, "data/human/KEGG_2021_Human.gmt")
]

svg_df = pd.read_csv(os.path.join(RES_DIR, "spagcn/all_domains_svg_summary.csv"))
anno_summary = []
fc_col = next((c for c in ['logfoldchanges', 'avg_log2FC', 'logfc'] if c in svg_df.columns), None)

for d in sorted(svg_df['target_domain'].unique()):
    subset = svg_df[(svg_df['target_domain'] == d) & (svg_df['pvals_adj'] < 0.05)]
    if fc_col: subset = subset[subset[fc_col] > 0.5]
    genes = subset.sort_values('pvals_adj').head(150)['genes'].tolist()

    if len(genes) < 5: continue

    enr = gp.enrichr(gene_list=genes, gene_sets=gmt_files, outdir=None)
    res = enr.results

    if not res.empty:
        sig_res = res[res['P-value'] < 0.05].copy()
        if not sig_res.empty:
            sig_res['domain'] = f"Domain_{d}"
            anno_summary.append(sig_res)

            # 准备绘图数据 (取 Top 10)
            plot_df = sig_res.sort_values('P-value').head(10).copy()
            plot_df['-log10P'] = -np.log10(plot_df['P-value'])

            # --- 优化绘图布局 ---
            # 1. 增加画布宽度 (14寸)，为右侧图例留出空间
            fig, ax = plt.subplots(figsize=(14, 8))

            # 2. 处理长文本：将过长的路径名换行显示
            plot_df['Term'] = [textwrap.fill(t, width=40) for t in plot_df['Term']]

            # 3. 绘制气泡图
            scatter = sns.scatterplot(data=plot_df, x='-log10P', y='Term',
                                      size='Overlap', hue='-log10P',
                                      palette='viridis', sizes=(100, 500),
                                      edgecolors='black', ax=ax)

            # 4. [核心修复] 将图例移动到绘图区外面
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

            plt.title(f"Domain {d} Functional Pathways", fontsize=15)
            plt.xlabel("-log10(P-value)", fontsize=12)
            plt.ylabel("")  # 隐藏左侧标签，因为 Term 已经很清晰了

            # 5. 自动调整边距，防止左侧文字太长被切掉
            plt.subplots_adjust(left=0.35, right=0.8)

            # 6. 保存时使用 bbox_inches='tight' 确保包含所有外部图例
            plt.savefig(os.path.join(OUT_DIR, f"03_GO_Domain_{d}.pdf"), bbox_inches='tight')
            plt.savefig(os.path.join(OUT_DIR, f"03_GO_Domain_{d}.png"), dpi=150, bbox_inches='tight')
            plt.close()
            print(f"    Domain {d}: Visualizations saved.")

# ===========================================================================
# [Module 5] 标志基因热图
# ===========================================================================
print("[Module 5] Plotting marker heatmap...")
markers = ['CD19', 'CD3D', 'MS4A1', 'IGHG1', 'CD8A']
valid = [m for m in markers if m in adata.var_names]
if valid:
    sc.pl.heatmap(adata, valid, groupby='domain', show=False, cmap='viridis')
    plt.savefig(os.path.join(OUT_DIR, "04_Markers.pdf"))
    plt.close()

print("Joint Analysis Finished.")