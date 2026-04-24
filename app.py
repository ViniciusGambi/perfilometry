#!/usr/bin/env python3
"""
Análise de Perfilometria - Aplicação Web (Streamlit)
Com visualização em tempo real e rodapé fixo
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(
    page_title="Perfilometria",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS para layout compacto
st.markdown("""
<style>
    .block-container {
        padding-top: 2.5rem !important;
        padding-bottom: 0 !important;
    }
    h1 {
        font-size: 1.4rem !important;
        margin-bottom: 0.3rem !important;
        margin-top: 0.5rem !important;
    }
    footer {display: none !important;}
    [data-testid="stMetricValue"] {
        font-size: 1rem !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.7rem !important;
    }
    .stNumberInput > div > div > input {
        padding: 0.3rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# FUNÇÕES DE PROCESSAMENTO
# ============================================================

def detect_vertical_grid_lines(img):
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
    grouped = []
    current = [v_lines[0]]
    for x in v_lines[1:]:
        if x - current[-1] <= 3:
            current.append(x)
        else:
            grouped.append(int(np.mean(current)))
            current = [x]
    grouped.append(int(np.mean(current)))
    filtered = [grouped[0]] if grouped else []
    for x in grouped[1:]:
        if x - filtered[-1] > 40:
            filtered.append(x)
    if len(filtered) >= 2:
        spacings = np.diff(filtered)
        median = np.median(spacings)
        valid = spacings[(spacings > median * 0.8) & (spacings < median * 1.2)]
        avg_spacing = np.mean(valid) if len(valid) > 0 else median
    else:
        avg_spacing = 0
    return filtered, avg_spacing

def detect_horizontal_grid_lines(img):
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
    grouped = []
    current = [h_lines[0]]
    for y in h_lines[1:]:
        if y - current[-1] <= 3:
            current.append(y)
        else:
            grouped.append(int(np.mean(current)))
            current = [y]
    grouped.append(int(np.mean(current)))
    filtered = [grouped[0]] if grouped else []
    for y in grouped[1:]:
        if y - filtered[-1] > 40:
            filtered.append(y)
    if len(filtered) >= 2:
        spacings = np.diff(filtered)
        median = np.median(spacings)
        valid = spacings[(spacings > median * 0.8) & (spacings < median * 1.2)]
        avg_spacing = np.mean(valid) if len(valid) > 0 else median
    else:
        avg_spacing = 0
    return filtered, avg_spacing

def calculate_scales(img):
    v_lines, spacing_x = detect_vertical_grid_lines(img)
    h_lines, spacing_y = detect_horizontal_grid_lines(img)
    mm_per_pixel_x = 0.5 / spacing_x if spacing_x > 0 else 0.008
    um_per_pixel_y = 0.5 / spacing_y if spacing_y > 0 else 0.008
    return {
        'mm_per_pixel_x': mm_per_pixel_x,
        'um_per_pixel_y': um_per_pixel_y,
        'spacing_x': spacing_x,
        'spacing_y': spacing_y
    }

def find_baseline(img):
    b, g, r = cv2.split(img)
    red_mask = ((r > 150) & (g < 120) & (b < 120)).astype(np.uint8) * 255
    row_counts = np.sum(red_mask > 0, axis=1)
    if np.max(row_counts) > 10:
        return np.argmax(row_counts), red_mask
    return int(img.shape[0] * 0.625), red_mask

def detect_green_profile(img, h_min, h_max, s_min, v_min):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([h_min, s_min, v_min]), np.array([h_max, 255, 255]))
    profile = {}
    height, width = mask.shape
    for x in range(width):
        col = mask[:, x]
        pts = np.where(col > 0)[0]
        if len(pts) > 0:
            profile[x] = {'y_top': pts.min(), 'y_bottom': pts.max(), 'y_center': int(np.median(pts))}
    return profile, mask

def paint_valley(img, profile, y_baseline, mask_green):
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
                valley_data.append({'x': x, 'y_baseline': y_baseline, 'y_green': y_green_top, 'delta_y': delta})
                for y in range(y_baseline + 1, y_green_top):
                    if 0 <= y < height and mask_green[y, x] == 0:
                        if np.any(mask_green[y:y_green_top+1, x] > 0):
                            alpha = 0.7
                            orig = result[y, x].astype(np.float32)
                            result[y, x] = (orig * (1 - alpha) + valley_color * alpha).astype(np.uint8)
                            valley_pixels += 1
    return result, valley_pixels, valley_data

def calculate_area(valley_data, mm_per_pixel_x, um_per_pixel_y):
    if not valley_data:
        return 0, 0
    total = sum(mm_per_pixel_x * seg['delta_y'] * um_per_pixel_y for seg in valley_data)
    return total, total * 1000

def cv2_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def pil_to_cv2(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# ============================================================
# INTERFACE PRINCIPAL
# ============================================================
st.title("📊 Perfilometria")

uploaded_file = st.file_uploader("Upload", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")

# ============================================================
# PROCESSAMENTO E EXIBIÇÃO
# ============================================================
if uploaded_file is not None:
    pil_image = Image.open(uploaded_file)
    img_original = pil_to_cv2(pil_image)
    height, width = img_original.shape[:2]
    
    crop_x_left, crop_x_right = 150, min(1560, width)
    crop_y_top, crop_y_bottom = 230, min(800, height)
    
    crop = img_original[crop_y_top:crop_y_bottom, crop_x_left:crop_x_right].copy()
    
    if crop.size > 0:
        # Pega valores do session_state (ou defaults)
        h_min = st.session_state.get('h_min', 60)
        h_max = st.session_state.get('h_max', 80)
        s_min = st.session_state.get('s_min', 137)
        v_min = st.session_state.get('v_min', 137)
        show_mask = st.session_state.get('show_mask', True)
        show_baseline = st.session_state.get('show_baseline', True)
        
        scales = calculate_scales(crop)
        mm_per_pixel_x = scales['mm_per_pixel_x']
        um_per_pixel_y = scales['um_per_pixel_y']
        
        y_baseline, _ = find_baseline(crop)
        profile, mask_green = detect_green_profile(crop, h_min, h_max, s_min, v_min)
        result, valley_pixels, valley_data = paint_valley(crop, profile, y_baseline, mask_green)
        area_mm_um, area_um2 = calculate_area(valley_data, mm_per_pixel_x, um_per_pixel_y)
        
        # Preparar imagens
        display_img = result.copy()
        if show_baseline:
            cv2.line(display_img, (0, y_baseline), (display_img.shape[1], y_baseline), (0, 0, 255), 2)
        
        mask_vis = np.zeros_like(crop)
        mask_vis[mask_green > 0] = [0, 255, 0]
        if show_baseline:
            cv2.line(mask_vis, (0, y_baseline), (mask_vis.shape[1], y_baseline), (0, 0, 255), 2)
        
        # Abas para alternar entre imagens
        tab1, tab2 = st.tabs(["Resultado", "Máscara Verde"])
        
        with tab1:
            st.image(cv2_to_pil(display_img), use_container_width=True)
        
        with tab2:
            st.image(cv2_to_pil(mask_vis), use_container_width=True)
        
        # Controles embaixo das abas
        ctrl = st.columns([1, 1, 1, 1, 0.5])
        with ctrl[0]:
            st.slider("H min", 0, 180, 60, key="h_min")
        with ctrl[1]:
            st.slider("H max", 0, 180, 80, key="h_max")
        with ctrl[2]:
            st.slider("S min", 0, 255, 137, key="s_min")
        with ctrl[3]:
            st.slider("V min", 0, 255, 137, key="v_min")
        with ctrl[4]:
            st.checkbox("Baseline", value=True, key="show_baseline")
        
        # Métricas em linha
        m_cols = st.columns([1, 1, 2, 1.5, 0.8])
        with m_cols[0]:
            st.metric("Área", f"{area_um2:.1f} μm²")
        with m_cols[1]:
            st.metric("Pixels", valley_pixels)
        
        if valley_data:
            depths = [seg['delta_y'] * um_per_pixel_y for seg in valley_data]
            x_pos = [seg['x'] for seg in valley_data]
            width_mm = (max(x_pos) - min(x_pos)) * mm_per_pixel_x if len(x_pos) > 1 else 0
            
            with m_cols[2]:
                st.metric("Prof (média/máx)", f"{np.mean(depths):.2f} / {max(depths):.2f} μm")
            with m_cols[3]:
                st.metric("Largura", f"{width_mm:.3f} mm")
        
        with m_cols[4]:
            buf = io.BytesIO()
            cv2_to_pil(result).save(buf, format='PNG')
            st.download_button("📥", buf.getvalue(), "resultado.png", "image/png")
        

else:
    st.info("👆 Faça upload de uma imagem para começar")
