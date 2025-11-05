import numpy as np

def generate_random_bits(
    length: int, 
    seed: int = 2025
) -> str:
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=length)
    return ''.join(map(str, bits))

def get_accuracy(embed: str, extract: str) -> float:
    assert len(embed) == len(extract), f'String lengths ({len(embed)} != {len(extract)}) do not match!'
    if len(embed) != len(extract):
        raise ValueError('String lengths do not match!')
    matches = sum(c1 == c2 for c1, c2 in zip(embed, extract))
    return matches / len(embed)

def generate_random_indices(
    choices: int,
    length: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    indices = rng.choice(choices, size=length, replace=False) # `replace=False` 要求每个索引只能选择一次
    return indices

def generate_udistribution(num: int, seed: int):
    rng = np.random.default_rng(seed)
    return rng.random(num)