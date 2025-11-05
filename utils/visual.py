from typing import List
import numpy as np
import cv2
import matplotlib.pyplot as plt

from algorithms.dct_2p import dct_2p_embed, dct_2p_extract
from utils.data import get_accuracy, generate_random_indices


def analyze_lsb(
    cover: np.ndarray,
    stego: np.ndarray,
    embed_bits: str,
    seed: int,
    save_path: str
):
    assert cover.shape == stego.shape, f'The shape of cover ({cover.shape}) and stego ({stego.shape}) does not match'
    assert cover.ndim == 2, f'Input cover must be 2D array, but got {cover.shape}'
    
    h, w = cover.shape
    indices = generate_random_indices(h * w, len(embed_bits), seed)

    # 像素差值
    pixel_diff = np.abs(cover.astype(int) - stego.astype(int))
    
    # 嵌入位置
    embed_map = np.zeros_like(cover, dtype=np.uint8)
    embed_map_flatten = embed_map.flatten()
    embed_map_flatten[indices] = 255
    embed_map = embed_map_flatten.reshape(cover.shape)
    
    # 灰度直方图
    cover_hist, bins = np.histogram(cover.flatten(), bins=256, range=(0, 255))
    stego_hist, _ = np.histogram(stego.flatten(), bins=256, range=(0, 255))
    
    # 相邻像素点差值
    cover_diff = cover_hist[0::2] - cover_hist[1::2]
    stego_diff = stego_hist[0::2] - stego_hist[1::2]
    
    # 绘图
    fig, axes = plt.subplots(4, 2, figsize=(12, 16))
    plt.subplots_adjust(hspace=0.3, wspace=0.2)
    
    # cover VS stego
    axes[0, 0].imshow(cover, cmap='gray')
    axes[0, 0].set_title('Cover')
    axes[0, 0].axis('off')
    axes[0, 1].imshow(stego, cmap='gray')
    axes[0, 1].set_title('Stego')
    axes[0, 1].axis('off')
    
    # 像素差值 VS 嵌入位置
    axes[1, 0].imshow(pixel_diff, cmap='gray')
    axes[1, 0].set_title('Pixel Difference')
    axes[1, 0].axis('off')
    axes[1, 1].imshow(embed_map, cmap='gray')
    axes[1, 1].set_title('Embedding Positions')
    axes[1, 1].axis('off')
    
    # cover 直方图 VS stego 直方图
    axes[2, 0].bar(bins[:-1], cover_hist, color='blue')
    axes[2, 0].set_title('Cover Histogram')
    ymin, ymax = axes[2, 0].get_ylim()
    axes[2, 1].bar(bins[:-1], stego_hist, color='green')
    axes[2, 1].set_title('Stego Histogram')
    axes[2, 1].set_ylim(ymin, ymax)
    
    # cover 相邻像素差值 VS stego 相邻像素差值
    axes[3, 0].plot(range(len(cover_diff)), cover_diff, color='blue')
    axes[3, 0].set_title('Cover Adjacent Bin Difference')
    ymin, ymax = axes[3, 0].get_ylim()
    axes[3, 1].plot(range(len(stego_diff)), stego_diff, color='green')
    axes[3, 1].set_title('Stego Adjacent Bin Difference')
    axes[3, 1].set_ylim(ymin, ymax)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

def analyze_dct_2p(
    cover: np.ndarray,
    stego: np.ndarray,
    embed_bits: str,
    block_size: int,
    jpeg_qualities: List[int],
    gaps: List[float],
    save_path: str
):
    # 像素差值
    pixel_diff = np.abs(cover.astype(int) - stego.astype(int))
    
    def block_dct(img: np.ndarray, block_size: int) -> np.ndarray:
        h, w = img.shape
        dct_blocks = np.zeros_like(img, dtype=np.float32)
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                block = img[i: i+block_size, j: j+block_size].astype(np.float32)
                dct_block = cv2.dct(block)
                dct_blocks[i: i+block_size, j: j+block_size] = dct_block
        return dct_blocks
    
    # DCT 系数差值（对数缩放）
    cover_dct = block_dct(cover, block_size)
    stego_dct = block_dct(stego, block_size)
    dct_diff = np.log(np.abs(cover_dct - stego_dct) + 1)
    
    accuracies = np.zeros((len(jpeg_qualities), len(gaps)))
    
    for i, q in enumerate(jpeg_qualities):
        for j, g in enumerate(gaps):
            temp = dct_2p_embed(cover, embed_bits, gap=g, block_size=block_size)
            enc_params = [cv2.IMWRITE_JPEG_QUALITY, q]
            _, enc_img = cv2.imencode('.jpg', temp, enc_params)
            jpeg_stego = cv2.imdecode(enc_img, cv2.IMREAD_GRAYSCALE)
            
            extract_bits = dct_2p_extract(jpeg_stego, len(embed_bits))
            acc = get_accuracy(embed_bits, extract_bits)
            accuracies[i, j] = acc
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 16))
    
    # cover VS stego
    axes[0, 0].imshow(cover, cmap='gray')
    axes[0, 0].set_title('Cover')
    axes[0, 0].axis('off')
    axes[0, 1].imshow(stego, cmap='gray')
    axes[0, 1].set_title('Stego')
    axes[0, 1].axis('off')
    
    # 像素值差异 VS DCT 系数差异
    axes[1, 0].imshow(pixel_diff, cmap='gray')
    axes[1, 0].set_title('Pixel Difference')
    axes[1, 0].axis('off')
    axes[1, 1].imshow(dct_diff, cmap='gray')
    axes[1, 1].set_title('DCT Difference (log scale)')
    axes[1, 1].axis('off')
    
    # 准确率 + JPEG 质量折线图
    fig.delaxes(axes[2, 1])
    for j, g in enumerate(gaps):
        axes[2, 0].plot(jpeg_qualities, accuracies[:, j], marker='o', label=f'gap={g}')
    axes[2, 0].set_xlabel('JPEG Quality')
    axes[2, 0].set_ylabel('Extraction Accuracy')
    axes[2, 0].set_title('Accuracy vs JPEG Quality for Different Gaps')
    axes[2, 0].grid(True)
    axes[2, 0].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)