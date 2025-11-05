import numpy as np
from utils.data import generate_random_indices, generate_udistribution
import logging
logger = logging.getLogger(__name__)


def do_embedding(
    pixel: np.uint8, 
    bit: str, 
    prob: float
) -> np.uint8:
    bit = int(bit)
    pixel = int(pixel)  # convert to Python int to avoid overflow
    lsb = pixel % 2

    if lsb == bit:
        return np.uint8(pixel)

    delta = 1 if prob >= 0.5 else -1

    if pixel == 0 and delta == -1:
        pixel += 1
    elif pixel == 255 and delta == 1:
        pixel -= 1
    else:
        pixel += delta

    return np.uint8(pixel)

def lsbm_rand_embed(
    cover: np.ndarray,
    msg_bits: str,
    seed: int
) -> np.ndarray:
    logger.debug(f'Start to do LSBM random embedding (seed = {seed})...')
    if cover.ndim != 2:
        raise ValueError(f'Input cover must be 2D array, but got shape {cover.shape}')
    stego_flatten = cover.flatten()
    num_pixels = stego_flatten.shape[0]
    if len(msg_bits) > num_pixels:
        raise ValueError(f'The number of input message bits ({len(msg_bits)}) exceeds the maximum embedding capacity of the cover ({num_pixels})')
    
    # 使用随机种子生成嵌入的像素点索引
    indices = generate_random_indices(num_pixels, len(msg_bits), seed)
    probs = generate_udistribution(len(msg_bits), seed)
    for idx, bit, prob in zip(indices, msg_bits, probs):
        new_pixel = do_embedding(stego_flatten[idx], bit, prob)
        logger.debug(f'Message bit index = {idx}, make pixel {stego_flatten[idx]} to {new_pixel}')
        stego_flatten[idx] = new_pixel

    stego = stego_flatten.reshape(cover.shape)
    logger.debug('LSBM random embedding is done')
    return stego

def lsbm_rand_extract(
    stego: np.ndarray,
    bit_length: int,
    seed: int
) -> str:
    logger.debug(f'Start to do LSBM random extraction (seed = {seed})...')
    stego_flatten = stego.flatten()
    num_pixels = stego_flatten.shape[0]
    
    indices = generate_random_indices(num_pixels, bit_length, seed)
    # 根据索引提取 stego 每个像素点的最后一位
    extract_bits = []
    for idx in indices:
        bit = stego_flatten[idx] & 0b00000001
        extract_bits.append(str(bit))
        logger.debug(f'Pixel index = {idx}, get the LSB = {bit}')
        
    logger.debug('LSBM random extraction is done')   
    return ''.join(extract_bits)