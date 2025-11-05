

def text2bin(text: str) -> str:
    return ''.join(format(byte, '08b') for byte in text.encode('utf-8'))

def bin2text(bin: str) -> str:
    assert len(bin) % 8 == 0, f'Binary string length ({len(bin)}) must be a multiple of 8'
    byte_arr = bytes(int(bin[i: i+8], 2) for i in range(0, len(bin), 8))
    return byte_arr.decode('utf-8')