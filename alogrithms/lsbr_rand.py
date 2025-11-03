import numpy as np

def lsbr_rand_embed(
    cover: np.ndarray,
    msg_bits: str,
    seed: int
) -> np.ndarray:
    if cover.ndim != 2:
        raise ValueError(f'Input cover must be 2D array, but got shape {cover.shape}')
    stego_flatten = cover.flatten()
    num_pixels = stego_flatten.shape[0]
    if len(msg_bits) > num_pixels:
        raise ValueError(f'The number of input message bits ({len(msg_bits)}) exceeds the maximum embedding capacity of the cover ({num_pixels})')
    
    # 使用随机种子生成嵌入的像素点索引
    rng = np.random.default_rng(seed)
    indices = rng.choice(num_pixels, size=len(msg_bits), replace=False) # `replace=False` 要求每个索引只能选择一次
    for idx, bit in zip(indices, msg_bits):
        stego_flatten[idx] = (stego_flatten[idx] & 0b11111110) | int(bit)
    stego = stego_flatten.view(cover.shape)
    
    return stego

def lbsr_rand_extract(
    stego: np.ndarray,
    bit_length: int,
    seed: int
) -> str:
    stego_flatten = stego.flatten()
    num_pixels = stego_flatten.shape[0]
    
    rng = np.random.default_rng(seed)
    indices = rng.choice(num_pixels, size=bit_length, replace=False)
    # 根据索引提取 stego 每个像素点的最后一位
    extract_bits = []
    for idx in indices:
        bit = stego_flatten[idx] & 0b00000001
        extract_bits.append(str(bit))
        
    return ''.join(extract_bits)