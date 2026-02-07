# ===========================================================================
# 文件名称: 31_visualize_xenium_mask.py
# 脚本功能: 可视化 Xenium 影像与 Cellpose 生成的 .npy 掩码叠加效果
# 作者: Kuroneko
# ---------------------------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from skimage import color, measure

# 1. 路径设置
BASE_DIR = r"F:\ST\code\data\seg"
OUT_DIR = r"F:\ST\code\results\seg"

# 输入文件
mask_path = os.path.join(OUT_DIR, "xenium_segmentation_masks.npy")
img_path = os.path.join(BASE_DIR, "Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_outs", "morphology_focus",
                        "morphology_focus_0003.ome.tif")

# 2. 加载数据 (使用内存映射模式防止爆内存)
print("正在加载影像与掩码...")
# 读取原图 (DAPI通道通常是第0通道)
img = tifffile.imread(img_path)
# 如果是多通道，只取第一个通道作为背景
if len(img.shape) > 2:
    img = img[0, :, :]

# 读取 .npy 掩码
mask = np.load(mask_path)

print(f"影像尺寸: {img.shape}")
print(f"掩码尺寸: {mask.shape}")


# 3. 定义裁剪函数 (只看局部，因为全图太大了看不清)
def plot_roi(image, mask, x_start, y_start, size=1000):
    """
    绘制局部区域的叠加图
    x_start, y_start: 切片的左上角坐标
    size: 切片大小 (像素)
    """
    x_end = min(x_start + size, image.shape[1])
    y_end = min(y_start + size, image.shape[0])

    # 裁剪
    img_crop = image[y_start:y_end, x_start:x_end]
    mask_crop = mask[y_start:y_end, x_start:x_end]

    # 归一化影像以便显示
    img_show = (img_crop - np.min(img_crop)) / (np.max(img_crop) - np.min(img_crop))

    # 生成边界线 (Outlines)
    # 将 mask 转换为边界线，这样不会遮挡细胞内部
    contours = measure.find_contours(mask_crop, 0.5)

    # 绘图
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_show, cmap='gray')

    # 叠加彩色掩码 (半透明)
    masked_overlay = np.ma.masked_where(mask_crop == 0, mask_crop)
    ax.imshow(masked_overlay, cmap='tab20', alpha=0.3, interpolation='none')

    # 叠加边界线 (更清晰)
    # for contour in contours:
    #     ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color='red')

    ax.set_title(f"ROI: X={x_start}, Y={y_start}")
    ax.axis('off')

    save_path = os.path.join(OUT_DIR, f"vis_mask_roi_x{x_start}_y{y_start}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  -> 已保存局部图: {save_path}")


# 4. 执行绘图 (切几个不同的位置看看)
print("正在生成局部细节图...")

# 选取图像中心的区域
center_x = img.shape[1] // 2
center_y = img.shape[0] // 2

# 绘制中心点
plot_roi(img, mask, center_x, center_y, size=800)

# 绘制左上角偏移一点的区域 (避开全黑背景)
plot_roi(img, mask, center_x + 2000, center_y + 2000, size=800)

print("可视化完成。请打开结果文件夹查看 .png 图片。")