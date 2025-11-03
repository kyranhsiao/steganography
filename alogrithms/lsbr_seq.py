import numpy as np

def lsbr_seq_embed(
    cover: np.ndarray,
    msg_bits: str
) -> np.ndarray:
    if cover.ndim != 2:
        raise ValueError(f'Input cover must be 2D array, but got shape {cover.shape}')
    stego_flatten = cover.flatten()
    num_pixels = stego_flatten.shape[0]
    if len(msg_bits) > num_pixels:
        raise ValueError(f'The number of input message bits ({len(msg_bits)}) exceeds the maximum embedding capacity of the cover ({num_pixels})')
    
    # 将 cover 像素点的最后一位替换为消息比特
    for idx, bit in enumerate(msg_bits):
        stego_flatten[idx] = (stego_flatten[idx] & 0b11111110) | int(bit)
    
    stego = stego_flatten.view(cover.shape)
    return stego

def lsbr_seq_extract(
    stego: np.ndarray,
    bit_length: int
) -> str:
    stego_flatten = stego.flatten()
    
    # 顺序提取 stego 每个像素点的最后一位
    extract_bits = []
    for idx in range(bit_length):
        bit = stego_flatten[idx] & 0b00000001
        extract_bits.append(str(bit))
        
    return ''.join(extract_bits)