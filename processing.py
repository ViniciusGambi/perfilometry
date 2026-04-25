"""
Funções de processamento de imagem para análise de perfilometria.
"""

import cv2
import numpy as np

from constants import (
    CROP_X_LEFT, CROP_X_RIGHT, CROP_Y_TOP, CROP_Y_BOTTOM,
    GREEN_H_MIN, GREEN_H_MAX, GREEN_S_MIN, GREEN_V_MIN,
    DEFAULT_MM_PER_GRID, DEFAULT_UM_PER_GRID
)


def detect_vertical_grid_lines(img):
    """
    Detecta linhas verticais da grade do gráfico.
    
    Returns:
        tuple: (lista de posições X das linhas, espaçamento médio em pixels)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    v_lines = []
    for x in range(width):
        col = gray[:, x]
        gray_pixels = np.sum((col > 160) & (col < 230))
        if gray_pixels > height * 0.3:
            v_lines.append(x)
    
    if not v_lines:
        return [], 0
    
    # Agrupa linhas próximas
    grouped = []
    current = [v_lines[0]]
    for x in v_lines[1:]:
        if x - current[-1] <= 3:
            current.append(x)
        else:
            grouped.append(int(np.mean(current)))
            current = [x]
    grouped.append(int(np.mean(current)))
    
    # Filtra linhas muito próximas
    filtered = [grouped[0]] if grouped else []
    for x in grouped[1:]:
        if x - filtered[-1] > 40:
            filtered.append(x)
    
    # Calcula espaçamento médio
    if len(filtered) >= 2:
        spacings = np.diff(filtered)
        median = np.median(spacings)
        valid = spacings[(spacings > median * 0.8) & (spacings < median * 1.2)]
        avg_spacing = np.mean(valid) if len(valid) > 0 else median
    else:
        avg_spacing = 0
    
    return filtered, avg_spacing


def detect_horizontal_grid_lines(img):
    """
    Detecta linhas horizontais da grade do gráfico.
    
    Returns:
        tuple: (lista de posições Y das linhas, espaçamento médio em pixels)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    h_lines = []
    for y in range(height):
        row = gray[y, :]
        gray_pixels = np.sum((row > 160) & (row < 230))
        if gray_pixels > width * 0.3:
            h_lines.append(y)
    
    if not h_lines:
        return [], 0
    
    # Agrupa linhas próximas
    grouped = []
    current = [h_lines[0]]
    for y in h_lines[1:]:
        if y - current[-1] <= 3:
            current.append(y)
        else:
            grouped.append(int(np.mean(current)))
            current = [y]
    grouped.append(int(np.mean(current)))
    
    # Filtra linhas muito próximas
    filtered = [grouped[0]] if grouped else []
    for y in grouped[1:]:
        if y - filtered[-1] > 40:
            filtered.append(y)
    
    # Calcula espaçamento médio
    if len(filtered) >= 2:
        spacings = np.diff(filtered)
        median = np.median(spacings)
        valid = spacings[(spacings > median * 0.8) & (spacings < median * 1.2)]
        avg_spacing = np.mean(valid) if len(valid) > 0 else median
    else:
        avg_spacing = 0
    
    return filtered, avg_spacing


def calculate_scales(img, mm_per_grid=DEFAULT_MM_PER_GRID, um_per_grid=DEFAULT_UM_PER_GRID):
    """
    Calcula a escala de conversão pixel→unidade real.
    
    Args:
        img: Imagem do gráfico recortado
        mm_per_grid: Milímetros por quadrado no eixo X
        um_per_grid: Micrômetros por quadrado no eixo Y
    
    Returns:
        dict: {'mm_per_pixel_x', 'um_per_pixel_y', 'spacing_x', 'spacing_y'}
    """
    v_lines, spacing_x = detect_vertical_grid_lines(img)
    h_lines, spacing_y = detect_horizontal_grid_lines(img)
    
    mm_per_pixel_x = mm_per_grid / spacing_x if spacing_x > 0 else 0.008
    um_per_pixel_y = um_per_grid / spacing_y if spacing_y > 0 else 0.008
    
    return {
        'mm_per_pixel_x': mm_per_pixel_x,
        'um_per_pixel_y': um_per_pixel_y,
        'spacing_x': spacing_x,
        'spacing_y': spacing_y
    }


def find_baseline(img):
    """
    Encontra a linha vermelha de referência (baseline).
    
    Returns:
        tuple: (posição Y da baseline, máscara vermelha)
    """
    b, g, r = cv2.split(img)
    red_mask = ((r > 150) & (g < 120) & (b < 120)).astype(np.uint8) * 255
    row_counts = np.sum(red_mask > 0, axis=1)
    
    if np.max(row_counts) > 10:
        return np.argmax(row_counts), red_mask
    
    return int(img.shape[0] * 0.625), red_mask


def detect_green_profile(img, h_min=GREEN_H_MIN, h_max=GREEN_H_MAX, s_min=GREEN_S_MIN, v_min=GREEN_V_MIN):
    """
    Detecta a linha verde do perfil.
    
    Args:
        img: Imagem do gráfico
        h_min, h_max: Faixa de matiz (Hue) para verde
        s_min: Saturação mínima
        v_min: Valor (brilho) mínimo
    
    Returns:
        tuple: (dicionário do perfil {x: {y_top, y_bottom, y_center}}, máscara verde)
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([h_min, s_min, v_min]), np.array([h_max, 255, 255]))
    
    profile = {}
    height, width = mask.shape
    
    for x in range(width):
        col = mask[:, x]
        pts = np.where(col > 0)[0]
        if len(pts) > 0:
            profile[x] = {
                'y_top': pts.min(),
                'y_bottom': pts.max(),
                'y_center': int(np.median(pts))
            }
    
    return profile, mask


def paint_valley(img, profile, y_baseline, mask_green):
    """
    Pinta o vale entre a baseline e o perfil verde.
    
    Args:
        img: Imagem do gráfico
        profile: Dicionário do perfil verde
        y_baseline: Posição Y da baseline
        mask_green: Máscara do perfil verde
    
    Returns:
        tuple: (imagem pintada, quantidade de pixels pintados, dados do vale)
    """
    result = img.copy()
    height, width = img.shape[:2]
    valley_color = np.array([255, 100, 0], dtype=np.float32)
    
    valley_pixels = 0
    valley_data = []
    
    for x, data in sorted(profile.items()):
        y_green_top = data['y_top']
        
        if y_green_top > y_baseline + 1:
            delta = y_green_top - y_baseline - 1
            
            if delta > 0:
                valley_data.append({
                    'x': x,
                    'y_baseline': y_baseline,
                    'y_green': y_green_top,
                    'delta_y': delta
                })
                
                for y in range(y_baseline + 1, y_green_top):
                    if 0 <= y < height and mask_green[y, x] == 0:
                        if np.any(mask_green[y:y_green_top+1, x] > 0):
                            alpha = 0.7
                            orig = result[y, x].astype(np.float32)
                            result[y, x] = (orig * (1 - alpha) + valley_color * alpha).astype(np.uint8)
                            valley_pixels += 1
    
    return result, valley_pixels, valley_data


def calculate_area(valley_data, mm_per_pixel_x, um_per_pixel_y):
    """
    Calcula a área do vale.
    
    Args:
        valley_data: Lista de segmentos do vale
        mm_per_pixel_x: Escala X (mm/pixel)
        um_per_pixel_y: Escala Y (μm/pixel)
    
    Returns:
        tuple: (área em mm·μm, área em μm²)
    """
    if not valley_data:
        return 0, 0
    
    total = sum(mm_per_pixel_x * seg['delta_y'] * um_per_pixel_y for seg in valley_data)
    return total, total * 1000


def crop_graph(img):
    """
    Recorta a região do gráfico da imagem original.
    
    Args:
        img: Imagem original
    
    Returns:
        Imagem recortada
    """
    height, width = img.shape[:2]
    x_right = min(CROP_X_RIGHT, width)
    y_bottom = min(CROP_Y_BOTTOM, height)
    
    return img[CROP_Y_TOP:y_bottom, CROP_X_LEFT:x_right].copy()


def _evaluate_params(crop, y_baseline, mm_per_pixel_x, um_per_pixel_y, h_min, h_max, s_min, v_min):
    """Avalia uma combinação de parâmetros e retorna a área."""
    if h_max <= h_min:
        return 0
    
    profile, mask_green = detect_green_profile(crop, h_min, h_max, s_min, v_min)
    if not profile:
        return 0
    
    _, valley_pixels, valley_data = paint_valley(crop, profile, y_baseline, mask_green)
    _, area_um2 = calculate_area(valley_data, mm_per_pixel_x, um_per_pixel_y)
    
    return area_um2


def auto_adjust_parameters(crop, y_baseline, mm_per_pixel_x, um_per_pixel_y):
    """
    Otimiza parâmetros HSV usando Grid Search em duas fases:
    
    Fase 1 (grosso): Testa combinações com passo grande para achar região
    Fase 2 (fino): Refina com passo de 1 ao redor do melhor
    
    Args:
        crop: Imagem recortada do gráfico
        y_baseline: Posição Y da baseline
        mm_per_pixel_x: Escala X
        um_per_pixel_y: Escala Y
    
    Returns:
        dict: Melhores parâmetros {'h_min', 'h_max', 's_min', 'v_min', 'area', 'pixels'}
    """
    
    def evaluate(h_min, h_max, s_min, v_min):
        return _evaluate_params(crop, y_baseline, mm_per_pixel_x, um_per_pixel_y, h_min, h_max, s_min, v_min)
    
    # ========================================
    # FASE 1: Grid Search Grosso
    # ========================================
    h_min_coarse = [40, 50, 60, 70]
    h_max_coarse = [70, 80, 90, 100]
    s_min_coarse = [60, 100, 140, 180]
    v_min_coarse = [60, 100, 140, 180]
    
    best_params = {
        'h_min': GREEN_H_MIN,
        'h_max': GREEN_H_MAX,
        's_min': GREEN_S_MIN,
        'v_min': GREEN_V_MIN
    }
    best_area = 0
    
    for h_min in h_min_coarse:
        for h_max in h_max_coarse:
            if h_max <= h_min:
                continue
            for s_min in s_min_coarse:
                for v_min in v_min_coarse:
                    area = evaluate(h_min, h_max, s_min, v_min)
                    if area > best_area:
                        best_area = area
                        best_params = {
                            'h_min': h_min,
                            'h_max': h_max,
                            's_min': s_min,
                            'v_min': v_min
                        }
    
    # ========================================
    # FASE 2: Refinamento com coordinate descent (passo de 1)
    # ========================================
    # Muito mais rápido: otimiza um parâmetro por vez
    
    params = best_params.copy()
    improved = True
    
    while improved:
        improved = False
        
        for param_name in ['h_min', 'h_max', 's_min', 'v_min']:
            # Define limites
            if param_name == 'h_min':
                min_val, max_val = 30, 75
            elif param_name == 'h_max':
                min_val, max_val = 65, 110
            else:
                min_val, max_val = 30, 220
            
            # Tenta diminuir
            while params[param_name] > min_val:
                params[param_name] -= 1
                if param_name == 'h_min' and params['h_max'] <= params['h_min']:
                    params[param_name] += 1
                    break
                area = evaluate(params['h_min'], params['h_max'], params['s_min'], params['v_min'])
                if area > best_area:
                    best_area = area
                    best_params = params.copy()
                    improved = True
                else:
                    params[param_name] += 1
                    break
            
            # Tenta aumentar
            while params[param_name] < max_val:
                params[param_name] += 1
                if param_name == 'h_max' and params['h_max'] <= params['h_min']:
                    params[param_name] -= 1
                    break
                area = evaluate(params['h_min'], params['h_max'], params['s_min'], params['v_min'])
                if area > best_area:
                    best_area = area
                    best_params = params.copy()
                    improved = True
                else:
                    params[param_name] -= 1
                    break
    
    # Calcula pixels finais
    profile, mask_green = detect_green_profile(
        crop, best_params['h_min'], best_params['h_max'], best_params['s_min'], best_params['v_min']
    )
    _, valley_pixels, _ = paint_valley(crop, profile, y_baseline, mask_green)
    
    return {
        'h_min': best_params['h_min'],
        'h_max': best_params['h_max'],
        's_min': best_params['s_min'],
        'v_min': best_params['v_min'],
        'area': best_area,
        'pixels': valley_pixels
    }
