# ===========================================================================
# File Name: run_CellTrek_analysis_v3.3.R
# Author: Kuroneko
# Date: 2026-02-02
# Version: V3.3 (RCTD Data Source + Seurat v5 Reconstruction Fix)
#
# Description:
#     [CellTrek 单细胞空间映射全流程 - 官方推荐重构版]
#     
#     本脚本改用之前 RCTD 分析通过的原始数据 (Raw Data)，避免了旧 RDS 文件的兼容性问题。
#     针对 "subscript out of bounds" 报错，采用了 GitHub Issue 中 Seurat 官方推荐的
#     "Reconstruction" (重构) 方案。
#
#     核心修复步骤:
#     1. [Data Source Switch]: 使用 10x_brain 和 matrix.txt 作为输入。
#     2. [Extract]: 直接从 Assay5 对象中提取 counts 稀疏矩阵。
#     3. [Rebuild]: 使用 CreateAssayObject(counts = ...) 创建标准 v3 对象。
#     4. [Replace]: 将新对象覆盖回原 Seurat 对象中，彻底解决兼容性问题。
#     5. [Image Sync]: 修复重命名后的图像坐标同步问题。
#
#     功能模块:
#     1. [Data Loading]: 加载原始数据，预处理并执行重构修复。
#     2. [Integration]: ST 和 SC 共嵌入 (traint)。
#     3. [Mapping]: 单细胞空间映射 (celltrek)。
#     4. [SColoc]: 空间共定位分析。
#     5. [Scoexp]: 空间共表达分析。
#     6. [K-distance]: 空间距离计算。
#     7. [Export]: 保存所有结果。
#
# Input Files (RCTD Source Data):
#     - F:/ST/code/05/10x_brain/ (空间数据文件夹)
#     - F:/ST/code/05/sc/brain_sc_expression_matrix.txt (单细胞矩阵)
#     - F:/ST/code/05/sc/brain_sc_metadata.csv (单细胞元数据)
#
# Output Files (in F:/ST/code/05/CellTrek_RCTD_Output/):
#     - celltrek_object.rds
#     - scoloc_mst_matrix.csv
#     - kdist_results.csv
#     - scoexp_gene_modules.csv
#     - 01_raw_spatial_dimplot.png (原始空间数据分布)
#     - 02_raw_sc_dimplot.png (原始单细胞 UMAP)
#     - 03_coembedding_umap.png (ST-SC 共嵌入图)
#     - 04_celltrek_mapping_vis.pdf (CellTrek 映射结果)
#     - 05_scoloc_network.pdf (细胞类型共定位网络)
#     - 06_scoexp_heatmap.png (空间基因模块热图)
#     - 07_scoexp_module_spatial.png (基因模块空间分布)
#     - 08_kdistance_boxplot.png (细胞空间距离统计)
# ===========================================================================


# -------------------------------------------------------------------------
# [Module 1] 环境初始化
# -------------------------------------------------------------------------
message("[Module 1] Initializing Environment...")

Sys.setenv(LANGUAGE = "en")
options(stringsAsFactors = FALSE)
options(Seurat.object.assay.version = "v3")

rm(list=ls())

packages <- c("CellTrek", "dplyr", "Seurat", "viridis", "ggplot2", 
              "patchwork", "readr", "magrittr", "data.table", "Matrix", "pheatmap", "ggpubr")

for(pkg in packages){
  if(!require(pkg, character.only = TRUE)){
    stop(paste0("[Error] Package '", pkg, "' is not installed."))
  } else {
    message(paste0("    [OK] Loaded: ", pkg))
  }
}

# -------------------------------------------------------------------------
# [Module 2] 数据加载
# -------------------------------------------------------------------------
message("\n[Module 2] Loading RCTD Source Data...")

BASE_DIR <- "F:/ST/code/05/"
OUT_DIR <- file.path(BASE_DIR, "CellTrek_RCTD_Output")
if(!dir.exists(OUT_DIR)) dir.create(OUT_DIR)
setwd(OUT_DIR)

# 1. 路径定义
sc_exp_path <- file.path(BASE_DIR, "sc/brain_sc_expression_matrix.txt")
sc_meta_path <- file.path(BASE_DIR, "sc/brain_sc_metadata.csv")
spatial_dir <- file.path(BASE_DIR, "10x_brain")

if(!file.exists(sc_exp_path)) stop("SC Matrix not found.")
if(!dir.exists(spatial_dir)) stop("Spatial directory not found.")

# 2. 加载单细胞
message("    Reading SC Matrix...")
sc_exp <- fread(sc_exp_path, data.table = FALSE)
rownames(sc_exp) <- sc_exp[,1]
sc_exp <- sc_exp[,-1]
sc_mat <- as(as.matrix(sc_exp), "sparseMatrix")

message("    Reading SC Metadata...")
sc_meta <- read.csv(sc_meta_path)
rownames(sc_meta) <- sc_meta$X

message("    Creating SC Object...")
brain_sc <- CreateSeuratObject(counts = sc_mat, meta.data = sc_meta, min.cells = 3, min.features = 200)

# 3. 加载空间数据
message("    Loading Spatial Data...")
brain_st <- Load10X_Spatial(spatial_dir)

# =========================================================================
# [Fix 1] Assay 重构函数
# =========================================================================
reconstruct_assay_v3 <- function(obj, label) {
  assay_name <- DefaultAssay(obj)
  message(paste0("    [Fix 1] Reconstructing Assay '", assay_name, "' for ", label, "..."))
  
  counts_matrix <- tryCatch({
    LayerData(obj, assay = assay_name, layer = "counts")
  }, error = function(e) {
    GetAssayData(obj, assay = assay_name, slot = "counts")
  })
  
  new_assay <- CreateAssayObject(counts = counts_matrix)
  obj[[assay_name]] <- new_assay
  return(obj)
}

# =========================================================================
# [Fix 2] Image 深度修复函数
# =========================================================================
downgrade_visium_v2_to_v1_robust <- function(obj, label) {
  message(paste0("    [Fix 2] Checking Image Classes for ", label, "..."))
  
  img_names <- Images(obj)
  for (img_name in img_names) {
    curr_img <- obj[[img_name]]
    
    if (inherits(curr_img, "VisiumV2")) {
      message(paste0("      -> Detected VisiumV2: ", img_name, ". Downgrading..."))
      
      coords_raw <- GetTissueCoordinates(obj, image = img_name)
      coords_final <- data.frame(row.names = rownames(coords_raw))
      
      # 映射 x/y
      coords_final$imagecol <- if("x" %in% colnames(coords_raw)) coords_raw$x else coords_raw[,1]
      coords_final$imagerow <- if("y" %in% colnames(coords_raw)) coords_raw$y else coords_raw[,2]
      
      # 补全
      coords_final$tissue <- 1
      coords_final$row <- if("row" %in% colnames(coords_raw)) coords_raw$row else coords_final$imagerow
      coords_final$col <- if("col" %in% colnames(coords_raw)) coords_raw$col else coords_final$imagecol
      
      # 计算半径
      raw_scales <- curr_img@scale.factors
      raw_radius <- tryCatch({
        if (!is.null(raw_scales$spot) && !is.null(raw_scales$lowres)) {
          raw_scales$spot * raw_scales$lowres
        } else { 0.02 }
      }, error = function(e) { 0.02 })
      if(is.na(raw_radius) || length(raw_radius)==0) raw_radius <- 0.02
      
      # 构建 V1
      v1_img <- new(
        Class = "VisiumV1",
        image = curr_img@image,
        scale.factors = raw_scales,
        coordinates = coords_final,
        spot.radius = raw_radius,
        assay = DefaultAssay(obj),
        key = curr_img@key
      )
      
      obj[[img_name]] <- v1_img
      message("      -> Image successfully converted to VisiumV1.")
    }
  }
  return(obj)
}

# -------------------------------------------------------------------------
# [Module 3] 执行修复与预处理
# -------------------------------------------------------------------------
message("\n[Module 3] Executing Repairs & Preprocessing...")

brain_sc <- reconstruct_assay_v3(brain_sc, "SC Data")
brain_st <- reconstruct_assay_v3(brain_st, "ST Data")
brain_st <- downgrade_visium_v2_to_v1_robust(brain_st, "ST Data")

# 预处理
message("    Running Normalize/PCA/UMAP...")
brain_sc <- NormalizeData(brain_sc, verbose=FALSE) %>% 
  FindVariableFeatures(verbose=FALSE) %>% 
  ScaleData(verbose=FALSE) %>% 
  RunPCA(verbose=FALSE) %>% 
  RunUMAP(dims=1:30, verbose=FALSE)

brain_st <- NormalizeData(brain_st, verbose=FALSE) %>% 
  FindVariableFeatures(verbose=FALSE) %>% 
  ScaleData(verbose=FALSE) %>% 
  RunPCA(verbose=FALSE)

# 重命名与同步 (CRITICAL FIX HERE)
message("    Syncing coordinates...")
brain_sc <- RenameCells(brain_sc, new.names = make.names(Cells(brain_sc)))
brain_st <- RenameCells(brain_st, new.names = make.names(Cells(brain_st)))

img_name <- Images(brain_st)[1]
if(inherits(brain_st[[img_name]], "VisiumV1")) {
  if(nrow(brain_st[[img_name]]@coordinates) == ncol(brain_st)) {
    # [Fix] 提取 -> 修改 -> 赋值回对象
    img_obj <- brain_st[[img_name]]
    rownames(img_obj@coordinates) <- Cells(brain_st)
    brain_st[[img_name]] <- img_obj # <--- 这步必须有！
    message("    -> Coordinates synced successfully (Assigned back).")
  }
}

# [Plot 1 & 2] 鲁棒绘图
message("    Generating Plots 01 & 02...")

# 尝试使用 Seurat 画图，如果报错则使用 ggplot2 手动画
p1 <- tryCatch({
  SpatialDimPlot(brain_st) + ggtitle("ST Data (Fixed)")
}, error = function(e) {
  message("    [Warning] SpatialDimPlot failed. Using fallback ggplot.")
  # 手动提取坐标画图
  coords <- GetTissueCoordinates(brain_st, image = img_name)
  ggplot(coords, aes(x = imagecol, y = imagerow)) +
    geom_point(size = 1, color = "red") +
    scale_y_reverse() + # Visium y轴通常是反的
    theme_void() +
    ggtitle("ST Data (Fallback Plot)")
})
ggsave("01_raw_spatial_dimplot.png", p1, width = 6, height = 6)

plot_group <- if("Class" %in% colnames(brain_sc@meta.data)) "Class" else NULL
p2 <- DimPlot(brain_sc, group.by = plot_group) + ggtitle("SC Data (Fixed)") 
ggsave("02_raw_sc_dimplot.png", p2, width = 8, height = 6)

# -------------------------------------------------------------------------
# [Module 4] CellTrek 核心 (Mapping)
# -------------------------------------------------------------------------
message("\n[Module 4] Running CellTrek Core...")

cell_type_col <- "Class" 
if(!cell_type_col %in% colnames(brain_sc@meta.data)) {
  brain_sc$cell_type <- Idents(brain_sc)
  cell_type_col <- "cell_type"
}

# 1. Traint
message("    Running traint...")
brain_traint <- CellTrek::traint(st_data = brain_st, 
                                 sc_data = brain_sc, 
                                 sc_assay = "RNA", 
                                 cell_names = cell_type_col)

# [Plot 3]
p3 <- DimPlot(brain_traint, group.by = "type") + ggtitle("ST-SC Co-embedding")
ggsave("03_coembedding_umap.png", p3, width = 8, height = 8)

# 2. Celltrek
message("    Running celltrek (Mapping)...")
celltrek_res <- CellTrek::celltrek(st_sc_int = brain_traint, 
                                   int_assay = "traint", 
                                   sc_data = brain_sc, 
                                   sc_assay = "RNA", 
                                   reduction = "pca", 
                                   intp = TRUE, 
                                   intp_pnt = 5000, 
                                   intp_lin = FALSE, 
                                   nPCs = 30, 
                                   ntree = 1000, 
                                   dist_thresh = 0.55, 
                                   top_spot = 5, 
                                   spot_n = 5, 
                                   repel_r = 20, 
                                   repel_iter = 20, 
                                   keep_model = TRUE)

saveRDS(celltrek_res$celltrek, "celltrek_object.rds")

# [Plot 4]
message("    Generating Plot 04...")
ct_obj <- celltrek_res$celltrek
ct_obj$cell_type <- factor(ct_obj$cell_type, levels = sort(unique(ct_obj$cell_type)))

pdf("04_celltrek_mapping_vis.pdf", width = 10, height = 10)
CellTrek::celltrek_vis(ct_obj@meta.data %>% dplyr::select(coord_x, coord_y, cell_type:id_new),
                       ct_obj@images[[1]]@image, 
                       ct_obj@images[[1]]@scale.factors$lowres)
dev.off()

# -------------------------------------------------------------------------
# [Module 5] 高级分析与绘图
# -------------------------------------------------------------------------
message("\n[Module 5] Advanced Analysis...")

# 5.1 SColoc
message("    Running SColoc...")
top_cells <- names(sort(table(ct_obj$cell_type), decreasing = T))[1:min(10, length(unique(ct_obj$cell_type)))]
ct_sub <- subset(ct_obj, subset = cell_type %in% top_cells)
ct_sub$cell_type <- factor(ct_sub$cell_type, levels = top_cells)

sgraph <- CellTrek::scoloc(ct_sub, col_cell='cell_type', use_method='KL', eps=1e-50)
write.csv(as.matrix(sgraph$mst_cons), "scoloc_mst_matrix.csv")

# [Plot 5]
pdf("05_scoloc_network.pdf", width = 8, height = 8)
ct_count <- data.frame(freq = table(ct_obj$cell_type))
ct_class <- data.frame(id = levels(ct_sub$cell_type))
ct_class_new <- merge(ct_class, ct_count, by.x ="id", by.y = "freq.Var1")
CellTrek::scoloc_vis(sgraph$mst_cons, meta_data=ct_class_new)
dev.off()

# 5.2 Scoexp
message("    Running Scoexp...")
target_type <- top_cells[1]
ct_l5 <- subset(ct_obj, subset = cell_type == target_type)

# 确保子集也是 v3
if(!"counts" %in% slotNames(ct_l5[["RNA"]])) ct_l5 <- reconstruct_assay_v3(ct_l5, "Subset")
if(nrow(ct_l5@assays$RNA@scale.data) == 0) ct_l5@assays$RNA@scale.data <- matrix(0, 0, 0)

ct_l5 <- FindVariableFeatures(ct_l5, verbose=FALSE)
scoexp_res <- CellTrek::scoexp(celltrek_inp=ct_l5, assay='RNA', approach='cc', 
                               gene_select = VariableFeatures(ct_l5)[1:1000], 
                               sigm=140, avg_cor_min=.4, zero_cutoff=3, min_gen=10, max_gen=400)

if(length(scoexp_res$gs) > 0) {
  # [Plot 6]
  png("06_scoexp_heatmap.png", width=2000, height=2000, res=300)
  pheatmap::pheatmap(scoexp_res$wcor, show_rownames=F, show_colnames=F)
  dev.off()
  
  # [Plot 7]
  ct_l5 <- AddModuleScore(ct_l5, features=scoexp_res$gs, name='Mod_')
  mod_cols <- grep('Mod_', colnames(ct_l5@meta.data), value=TRUE)
  plot_list <- list()
  for(mod in mod_cols) {
    plot_list[[mod]] <- ggplot(ct_l5@meta.data, aes(x=coord_x, y=coord_y, color=.data[[mod]])) +
      geom_point(size=0.5) + scale_color_viridis() + theme_void() + ggtitle(mod) + coord_fixed()
  }
  ggsave("07_scoexp_module_spatial.png", wrap_plots(plot_list, ncol=min(3, length(plot_list))), width=12, height=6)
  write.csv(as.data.frame(summary(scoexp_res$gs)), "scoexp_gene_modules.csv")
} else {
  message("    [Warning] No gene modules found. Skipping Plot 06/07.")
}

# 5.3 K-distance
message("    Running K-distance...")
inp_df <- ct_obj@meta.data %>% dplyr::select(cell_names = cell_type, coord_x, coord_y)
kdist_res <- CellTrek::kdist(inp_df = inp_df, ref = target_type, ref_type = 'all', 
                             que = unique(inp_df$cell_names), k = 10, 
                             new_name = paste0("Dist_to_", target_type), keep_nn = FALSE)

write.csv(kdist_res$kdist_df, "kdist_results.csv")

# [Plot 8]
dist_df <- kdist_res$kdist_df
dist_df$cell_type <- rownames(dist_df)
actual_dist_col <- grep("Dist_to", colnames(dist_df), value = TRUE)[1]

p_dist <- ggboxplot(data = dist_df, x = "cell_type", y = actual_dist_col, 
                    fill = "cell_type", title = paste("Distance to", target_type)) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
ggsave("08_kdistance_boxplot.png", p_dist, width=10, height=6)

message("="*60)
message("CellTrek Analysis Completed Successfully.")
message("Results saved to: ", OUT_DIR)
message("="*60)