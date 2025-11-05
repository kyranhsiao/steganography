from typing import Tuple
import numpy as np
import cv2
import logging
logger = logging.getLogger(__name__)

def pairing_adjust(
    bit: str,
    p1: float,
    p2: float,
    gap: float
) -> Tuple[float, float]:
    d = p1 - p2
    
    # 根据消息比特对两点的间距进行最小化调整
    if bit == '1':
        if d >= gap:
            return p1, p2
        delta = (gap - d) / 2.0
        return p1 + delta, p2 - delta
    elif bit == '0':
        if d <= -gap:
            return p1, p2
        delta = (gap + d) / 2.0
        return p1 - delta, p2 + delta
    else:
        raise ValueError(f'Bit must be 0 or 1 but got {bit}')
        

def dct_2p_embed(
    cover: np.ndarray,
    msg_bits: str,
    gap: float,
    block_size: int = 8,
    coeff_pos: Tuple[Tuple[int, int], Tuple[int, int]] = ((2, 3), (3, 2))
) -> np.ndarray:
    logger.debug(f'Start to do DCT two-point embedding...')
    assert cover.ndim == 2, f'Input cover must be 2D array, but got shape {cover.shape}'
    assert gap >= 0, f'`gap` must be positive, but got {gap}'
    assert all(0 <= x < block_size and 0 <= y < block_size for x, y in coeff_pos), f'Two point positions must be in the limit of {block_size}, but got {coeff_pos}'
    
    h, w = cover.shape
    
    num_blk_h = h // block_size
    num_blk_w = w // block_size
    assert num_blk_h * num_blk_w >= len(msg_bits), f'Not enough blocks ({num_blk_h * num_blk_w}) for {len(msg_bits)} bits'
    
    stego = cover.astype(np.float32).copy()
    # 按顺序先遍历行再遍历列
    idx = 0
    for y in range(num_blk_h):
        for x in range(num_blk_w):
            ex = x * block_size
            ey = y * block_size
            block = stego[ey: ey+block_size, ex: ex+block_size]
            
            block_dct = cv2.dct(block)
            (u1, v1), (u2, v2) = coeff_pos
            
            # 取 DCT 系数中的两点，嵌入消息比特
            p1 = block_dct[u1, v1]
            p2 = block_dct[u2, v2]
            
            new_p1, new_p2 = pairing_adjust(
                msg_bits[idx],
                p1, p2,
                gap
            )
            block_dct[u1, v1] = new_p1
            block_dct[u2, v2] = new_p2
            logger.debug(f'Message bit = {msg_bits[idx]}, make ({p1:.4f}, {p2:.4f}) -> ({new_p1:.4f}, {new_p2:.4f}), gap {np.abs(p1 - p2):.4f} -> {np.abs(new_p1 - new_p2):.4f}')
            
            # 使用 iDCT 转换到像素域 
            block_idct = cv2.idct(block_dct)
            stego[ey: ey+block_size, ex: ex+block_size] = block_idct
            idx += 1
            
            if idx >= len(msg_bits):
                break
        if idx >= len(msg_bits):
            break
            
    stego = np.clip(stego, 0, 255)
    logger.debug('DCT two-point embedding is done')
    return stego.astype(np.uint8)

def dct_2p_extract(
    stego: np.ndarray,
    bit_length: int,
    block_size: int = 8,
    coeff_pos: Tuple[Tuple[int, int], Tuple[int, int]] = ((2, 3), (3, 2))
) -> str:
    logger.debug(f'Start to do DCT two-point extraction...')
    assert all(0 <= x < block_size and 0 <= y < block_size for x, y in coeff_pos), f'Two point positions must be in the limit of {block_size}, but got {coeff_pos}'
    
    h, w = stego.shape
    stego = stego.astype(np.float32)
    
    num_blk_h = h // block_size
    num_blk_w = w // block_size
    assert num_blk_h * num_blk_w >= bit_length, f'Not enough blocks ({num_blk_h * num_blk_w}) for {bit_length} bits'
    
    extract_bits = []
    idx = 0
    for y in range(num_blk_h):
        for x in range(num_blk_w):
            ex = x * block_size
            ey = y * block_size
            block = stego[ey: ey+block_size, ex: ex+block_size]
            
            block_dct = cv2.dct(block)
            (u1, v1), (u2, v2) = coeff_pos
            
            # 取 DCT 系数中的两点，提取消息比特
            p1 = block_dct[u1, v1]
            p2 = block_dct[u2, v2]
            
            bit = '1' if (p1 > p2) else '0' # 根据大小关系提取消息比特，不需要 `gap` 的参与
            extract_bits.append(bit)
            logger.debug(f'Pair = ({p1:.4f}, {p2:.4f}), get message bit = {bit}')
            idx += 1
            
            if idx >= bit_length:
                break
        if idx >= bit_length:
            break
    logger.debug('DCT two-point extraction is done')
    return ''.join(extract_bits)