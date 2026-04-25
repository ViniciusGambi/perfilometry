"""
Funções utilitárias para conversão de imagens.
"""

import cv2
import numpy as np
from PIL import Image


def cv2_to_pil(img):
    """Converte imagem OpenCV (BGR) para PIL (RGB)."""
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def pil_to_cv2(img):
    """Converte imagem PIL (RGB) para OpenCV (BGR)."""
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
