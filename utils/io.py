import logging
import os
from typing import Union, Optional, List

import numpy as np
import cv2

def set_up_logging(
    log_level: int = logging.INFO,
    log_format: str = '[%(asctime)s] %(levelname)s - %(name)s: %(message)s',
    date_format: str = '%Y-%m-%d %H:%M:%S',
    log_filepath: Optional[str] = None,
):
    # 清除现有手柄
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    handlers = [logging.StreamHandler()]
    
    # 创建日志文件
    if log_filepath is not None:
        os.makedirs(os.path.dirname(log_filepath) or '.', exist_ok=True)
        handlers.append(logging.FileHandler(log_filepath, encoding='utf-8'))
        
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=handlers
    )
    
    logging.debug('Logging has initialized.')
    
def load_gray_image(img_path: str) -> np.ndarray:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError('Input image is None')
    return img

def save_gray_image(
    img: np.ndarray,
    save_path: str,
    quality: int = 95,
    optimize: bool = True,
    progressive: bool = False,
) -> bool:
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    ext = os.path.splitext(save_path)[1].lower()
    params: List[int] = []
    
    if ext in ['.jpg', '.jpeg']:
        params = [
            cv2.IMWRITE_JPEG_QUALITY, quality,
            cv2.IMWRITE_JPEG_OPTIMIZE, int(optimize),
            cv2.IMWRITE_JPEG_PROGRESSIVE, int(progressive)
        ]

    success = cv2.imwrite(save_path, img, params)
    return success