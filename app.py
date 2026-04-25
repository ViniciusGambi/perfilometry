#!/usr/bin/env python3
"""
Análise de Perfilometria - Aplicação Web (Streamlit)
"""

import streamlit as st
import numpy as np
from PIL import Image
import io

from utils import cv2_to_pil, pil_to_cv2
from ocr import extract_axis_scale
from processing import crop_graph, process_image, auto_adjust_parameters, calculate_scales, find_baseline
from constants import (
    GREEN_H_MIN, GREEN_H_MAX, GREEN_S_MIN, GREEN_V_MIN,
    DEFAULT_MM_PER_GRID, DEFAULT_UM_PER_GRID
)

# ============================================================
# CONFIGURAÇÃO DO STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Perfilometria",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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
# INTERFACE
# ============================================================

st.title("📊 Perfilometria")

uploaded_file = st.file_uploader("Upload", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")

if uploaded_file is not None:
    # Carrega imagem
    pil_image = Image.open(uploaded_file)
    img_original = pil_to_cv2(pil_image)
    height, width = img_original.shape[:2]
    
    # OCR para detectar escala (só na primeira vez por arquivo)
    file_id = uploaded_file.name + str(uploaded_file.size)
    if 'last_file' not in st.session_state or st.session_state.last_file != file_id:
        st.session_state.last_file = file_id
        detected_mm, detected_um, ocr_debug = extract_axis_scale(img_original)
        st.session_state.mm_per_grid = detected_mm
        st.session_state.um_per_grid = detected_um
        st.session_state.ocr_debug = ocr_debug
    
    # Recorta gráfico
    crop = crop_graph(img_original)
    
    if crop.size > 0:
        # Inicializa valores no session_state se não existirem
        if 'h_min' not in st.session_state:
            st.session_state['h_min'] = GREEN_H_MIN
        if 'h_max' not in st.session_state:
            st.session_state['h_max'] = GREEN_H_MAX
        if 's_min' not in st.session_state:
            st.session_state['s_min'] = GREEN_S_MIN
        if 'v_min' not in st.session_state:
            st.session_state['v_min'] = GREEN_V_MIN
        if 'show_baseline' not in st.session_state:
            st.session_state['show_baseline'] = True
        
        # Pega valores do session_state
        h_min = st.session_state['h_min']
        h_max = st.session_state['h_max']
        s_min = st.session_state['s_min']
        v_min = st.session_state['v_min']
        show_baseline = st.session_state['show_baseline']
        mm_per_grid = st.session_state.get('mm_per_grid', DEFAULT_MM_PER_GRID)
        um_per_grid = st.session_state.get('um_per_grid', DEFAULT_UM_PER_GRID)
        
        # Botão de auto-ajuste (ANTES do processamento)
        st.divider()
        auto_col, _ = st.columns([1, 7])
        with auto_col:
            if st.button("🔧 Auto-ajuste"):
                with st.spinner("Otimizando..."):
                    scales = calculate_scales(crop, mm_per_grid, um_per_grid)
                    y_baseline, _ = find_baseline(crop)
                    best = auto_adjust_parameters(crop, y_baseline, scales['mm_per_pixel_x'], scales['um_per_pixel_y'])
                    st.session_state['h_min'] = best['h_min']
                    st.session_state['h_max'] = best['h_max']
                    st.session_state['s_min'] = best['s_min']
                    st.session_state['v_min'] = best['v_min']
                    h_min, h_max, s_min, v_min = best['h_min'], best['h_max'], best['s_min'], best['v_min']
        
        # Processamento principal
        result = process_image(crop, h_min, h_max, s_min, v_min, mm_per_grid, um_per_grid, show_baseline)
        
        # Abas de visualização
        tab1, tab2 = st.tabs(["Resultado", "Máscara Verde"])
        
        with tab1:
            st.image(cv2_to_pil(result['result_img']), use_container_width=True)
        
        with tab2:
            st.image(cv2_to_pil(result['mask_img']), use_container_width=True)
        
        # Controles
        ctrl = st.columns([1, 1, 1, 1, 1, 1, 0.5])
        
        with ctrl[0]:
            st.slider("H min", 0, 180, h_min, key="h_min")
        with ctrl[1]:
            st.slider("H max", 0, 180, h_max, key="h_max")
        with ctrl[2]:
            st.slider("S min", 0, 255, s_min, key="s_min")
        with ctrl[3]:
            st.slider("V min", 0, 255, v_min, key="v_min")
        with ctrl[4]:
            st.number_input("mm/quad (X)", value=mm_per_grid, step=0.1, format="%.1f", key="mm_per_grid")
        with ctrl[5]:
            st.number_input("μm/quad (Y)", value=um_per_grid, step=0.1, format="%.1f", key="um_per_grid")
        with ctrl[6]:
            st.checkbox("Baseline", value=True, key="show_baseline")
        
        # Métricas
        m_cols = st.columns([1, 1, 2, 1.5, 1.2, 0.8])
        
        with m_cols[0]:
            st.metric("Área", f"{result['area_um2']:.1f} μm²")
        with m_cols[1]:
            st.metric("Pixels", result['valley_pixels'])
        
        if result['depths']:
            with m_cols[2]:
                st.metric("Prof (média/máx)", f"{np.mean(result['depths']):.2f} / {max(result['depths']):.2f} μm")
            with m_cols[3]:
                st.metric("Largura", f"{result['width_mm']:.3f} mm")
        
        with m_cols[4]:
            st.metric("Área/quadrado", f"{result['grid_area_um2']:.1f} μm²")
        
        with m_cols[5]:
            buf = io.BytesIO()
            cv2_to_pil(result['result_raw']).save(buf, format='PNG')
            st.download_button("📥", buf.getvalue(), "resultado.png", "image/png")
        
        # Debug OCR
        if 'ocr_debug' in st.session_state:
            debug = st.session_state.ocr_debug
            with st.expander("📏 Ver escala do gráfico"):
                col_x, col_y = st.columns(2)
                with col_x:
                    if 'x_values' in debug:
                        st.write(f"**Valores X:** {debug['x_values'][:5]}... → **{st.session_state.mm_per_grid} mm/quad**")
                    if 'x_region' in debug:
                        st.image(debug['x_region'], use_container_width=True)
                with col_y:
                    if 'y_values' in debug:
                        st.write(f"**Valores Y:** {debug['y_values'][:5]}... → **{st.session_state.um_per_grid} μm/quad**")
                    if 'y_region' in debug:
                        st.image(debug['y_region'])

else:
    st.info("👆 Faça upload de uma imagem para começar")
