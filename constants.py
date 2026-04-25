"""
Constantes de configuração para análise de perfilometria.
"""

# ============================================================
# RECORTE DO GRÁFICO (em pixels)
# ============================================================
CROP_X_LEFT = 150
CROP_X_RIGHT = 1560
CROP_Y_TOP = 230
CROP_Y_BOTTOM = 800

# ============================================================
# REGIÕES DO OCR - EIXO X (em pixels)
# ============================================================
X_REGION_TOP = 770
X_REGION_BOTTOM = 810
X_REGION_LEFT = 140
X_REGION_RIGHT = 1550

# ============================================================
# REGIÕES DO OCR - EIXO Y (em pixels)
# ============================================================
Y_REGION_TOP = 230
Y_REGION_BOTTOM = 760
Y_REGION_LEFT = 160
Y_REGION_RIGHT = 218

# ============================================================
# DETECÇÃO DO VERDE (HSV) - VALORES PADRÃO
# ============================================================
GREEN_H_MIN = 53
GREEN_H_MAX = 84
GREEN_S_MIN = 118
GREEN_V_MIN = 76

# ============================================================
# ESCALAS PADRÃO (quando OCR falha)
# ============================================================
DEFAULT_MM_PER_GRID = 0.5
DEFAULT_UM_PER_GRID = 0.5
