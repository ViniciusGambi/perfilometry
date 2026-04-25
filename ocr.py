"""
Funções de OCR para detecção de escala dos eixos.
"""

import cv2
import re

from constants import (
    X_REGION_TOP, X_REGION_BOTTOM, X_REGION_LEFT, X_REGION_RIGHT,
    Y_REGION_TOP, Y_REGION_BOTTOM, Y_REGION_LEFT, Y_REGION_RIGHT,
    DEFAULT_MM_PER_GRID, DEFAULT_UM_PER_GRID
)

try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


def extract_axis_scale(img_original):
    """
    Usa OCR para extrair os números dos eixos X e Y e calcular a escala por quadrado.
    
    Args:
        img_original: Imagem original em formato OpenCV (BGR)
    
    Returns:
        tuple: (mm_per_grid, um_per_grid, debug_info)
    """
    debug = {}
    
    if not OCR_AVAILABLE:
        debug['status'] = 'Tesseract não disponível'
        return DEFAULT_MM_PER_GRID, DEFAULT_UM_PER_GRID, debug
    
    try:
        version = pytesseract.get_tesseract_version()
        debug['tesseract_version'] = str(version)
    except Exception as e:
        debug['status'] = f'Erro ao verificar Tesseract: {e}'
        return DEFAULT_MM_PER_GRID, DEFAULT_UM_PER_GRID, debug
    
    height, width = img_original.shape[:2]
    
    mm_per_grid = DEFAULT_MM_PER_GRID
    um_per_grid = DEFAULT_UM_PER_GRID
    
    # Recorta com proteção de limites
    x_t = max(0, X_REGION_TOP)
    x_b = min(height, X_REGION_BOTTOM)
    x_l = max(0, X_REGION_LEFT)
    x_r = min(width, X_REGION_RIGHT)
    
    y_t = max(0, Y_REGION_TOP)
    y_b = min(height, Y_REGION_BOTTOM)
    y_l = max(0, Y_REGION_LEFT)
    y_r = min(width, Y_REGION_RIGHT)
    
    x_region = img_original[x_t:x_b, x_l:x_r]
    y_region = img_original[y_t:y_b, y_l:y_r]
    
    # Salva as regiões para debug
    if x_region.size > 0:
        debug['x_region'] = cv2.cvtColor(x_region, cv2.COLOR_BGR2RGB)
    if y_region.size > 0:
        debug['y_region'] = cv2.cvtColor(y_region, cv2.COLOR_BGR2RGB)
    
    # OCR eixo X
    mm_per_grid = _extract_x_scale(x_region, debug)
    
    # OCR eixo Y
    um_per_grid = _extract_y_scale(y_region, debug)
    
    debug['status'] = 'OK'
    return mm_per_grid, um_per_grid, debug


def _extract_x_scale(x_region, debug):
    """Extrai escala do eixo X usando OCR."""
    mm_per_grid = DEFAULT_MM_PER_GRID
    
    try:
        x_gray = cv2.cvtColor(x_region, cv2.COLOR_BGR2GRAY)
        x_text = pytesseract.image_to_string(
            x_gray, 
            config='--psm 7 -c tessedit_char_whitelist=0123456789,.'
        )
        debug['x_raw_text'] = x_text.strip()
        
        # Procura padrões XX,X (números com vírgula como 19,0 19,5 20,0)
        x_numbers = re.findall(r'\d{1,2}[,\.]\d', x_text)
        x_values = []
        for n in x_numbers:
            try:
                x_values.append(float(n.replace(',', '.')))
            except:
                pass
        
        debug['x_values'] = x_values
        
        if len(x_values) >= 2:
            x_values = sorted(set(x_values))
            diffs = [round(x_values[i+1] - x_values[i], 2) for i in range(len(x_values)-1)]
            debug['x_diffs'] = diffs
            
            # Pega diferenças pequenas (entre 0.1 e 2.0) - típico de escalas
            valid_diffs = [d for d in diffs if 0.1 <= d <= 2.0]
            if valid_diffs:
                mm_per_grid = round(min(valid_diffs), 1)
                
    except Exception as e:
        debug['x_error'] = str(e)
    
    return mm_per_grid


def _extract_y_scale(y_region, debug):
    """Extrai escala do eixo Y usando OCR."""
    um_per_grid = DEFAULT_UM_PER_GRID
    
    try:
        y_gray = cv2.cvtColor(y_region, cv2.COLOR_BGR2GRAY)
        y_text = pytesseract.image_to_string(
            y_gray, 
            config='--psm 6 -c tessedit_char_whitelist=0123456789,.-'
        )
        debug['y_raw_text'] = y_text.strip()
        
        y_numbers = re.findall(r'-?\d+[,.]?\d*', y_text)
        y_values = []
        for n in y_numbers:
            try:
                y_values.append(float(n.replace(',', '.')))
            except:
                pass
        
        debug['y_values'] = y_values
        
        if len(y_values) >= 2:
            y_values = sorted(set(y_values))
            diffs = [round(abs(y_values[i+1] - y_values[i]), 2) for i in range(len(y_values)-1)]
            debug['y_diffs'] = diffs
            
            if diffs:
                valid = [d for d in diffs if d > 0]
                if valid:
                    um_per_grid = round(min(valid), 1)
                    
    except Exception as e:
        debug['y_error'] = str(e)
    
    return um_per_grid
