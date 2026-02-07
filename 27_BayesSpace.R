# ===========================================================================
# Project: Spatial Transcriptomics Analysis (Bioinformatics Platform)
# File Name: 24_b_spatial_domains_stagate.py
# Author: Kuroneko (PhD Student / AI-Bioinformatics Developer)
# Current Path: F:\ST\code
# GPU Environment: RTX 5070 (Mandatory for STAGATE GAT mechanism)
#
# Description:
#     [STAGATE 空间域划分与组织结构识别]
#     本脚本旨在利用图注意力自编码器 (STAGATE) 融合基因表达与空间位置信息。
#     相比于常规聚类，STAGATE 能够产生更平滑的组织边界，减少技术噪声。
#
# Workflow Summary:
#     1. 数据加载: 读取 Human Lymph Node Visium 数据。
#     2. 空间邻接图: 构建 KNN (k=6) 网络，定义 Spot 间的空间依赖。
#     3. 模型训练: 运行 GAT (Graph Attention Network) 提取 30 维隐变量。
#     4. 空间域识别: 对 STAGATE Embedding 进行 Leiden 聚类，划定组织区域。
#     5. 对比分析: 可视化对比 "纯表达聚类" 与 "空间融合聚类" 的优劣。
#
# Input Files:
#     - Main ST Data: F:\ST\code\data\V1_Human_Lymph_Node\ (H5 + Spatial info)
#     - Supplementary scRNA-seq: F:\ST\code\03\sc.h5ad (用于下游联合分析参考)
#
# Output Files:
#     - Visualization: F:\ST\code\results\stagate\plots\ (包含空间域分布对比图)
#     - Embeddings: F:\ST\code\results\stagate\stagate_embedding.csv (用于轨迹分析)
#     - Data Object: F:\ST\code\results\stagate\stagate_result.h5ad
# ===========================================================================

# [后续接 Module 1 导入库和环境设置代码...]

# 1. 环境准备
library(BayesSpace)
library(SpatialExperiment)
library(ggplot2)
library(patchwork)

set.seed(7788)
# 设置输入路径为 GBM1 数据集
input_path <- "F:/ST/code/04/GBM1_spaceranger_out/"

# 2. 数据读取与预处理
# 使用 readVisium 读取 SpaceRanger 输出的标准格式文件
sp1 <- readVisium(input_path)

# 标准化与降维：BayesSpace 基于 PCA 空间进行建模
sp1 <- scater::logNormCounts(sp1)
# spatialPreprocess 完成归一化、找高变基因 (HVGs) 和运行 PCA
# 算法理由：PCA 能去除噪声并保留生物学差异，是后续 MCMC 模型的基础
sp1 <- spatialPreprocess(sp1, platform = "Visium", n.PCs = 15, n.HVGs = 2000)

# 3. 确定聚类数 (q)
# qTune 运行不同 q 值下的似然评估，qPlot 辅助观察拐点
sp1 <- qTune(sp1, qs = seq(2, 10), platform = "Visium", d = 7)
qPlot(sp1)

# 4. 空间聚类 (Spatial Clustering)
# 算法理由：BayesSpace 假设相邻点的聚类标签更倾向于一致，通过 Potts 模型引入空间先验。
# 这比传统的单细胞 K-means 或 Leiden 更能识别具有解剖结构的组织区域。
sp1 <- spatialCluster(sp1, q = 7, platform = "Visium", d = 15,
                      init.method = "mclust", model = "t", gamma = 2,
                      nrep = 1000, burn.in = 100)

# 可视化原始点位聚类结果
clusterPlot(sp1) + labs(title = "BayesSpace Clustering (Spot Level)")

# 5. 分辨率增强 (Spatial Enhancement)
# 算法理由：Visium 的点 (55微米) 包含多个细胞。
# spatialEnhance 将每个 spot 进一步细分为更小的 subspots，并根据邻近表达推断其表达量。
sp1.enhanced <- spatialEnhance(sp1, q = 7, platform = "Visium", d = 15,
                               model = "t", gamma = 2, nrep = 1000, burn.in = 100)

# 增强特定基因的表达分布 (例如 PMEL)
# enhanceFeatures 使用 XGBoost 等模型将低分辨率的基因表达映射到高分辨率 subspots 上
markers <- c("PMEL", "COL1A1")
sp1.enhanced <- enhanceFeatures(sp1.enhanced, sp1, feature_names = markers)

# 对比原始与增强后的效果
p1 <- featurePlot(sp1, "PMEL") + labs(title = "Original")
p2 <- featurePlot(sp1.enhanced, "PMEL") + labs(title = "Enhanced")

p1 + p2
