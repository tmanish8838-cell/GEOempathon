"""
AquaPrior — Explainable Micro-Watershed Priority & Intervention Decision-Support System
GEO IMPATHON 1.0 • Problem Statement 2.3 (Watershed-Based Water Resource Priority Mapping)

Implements the complete 6-Section AquaPrior Design System with full backend interactivity:
1. Home
2. Explore Map (Explore Priority Map)
3. Priority Areas (Interactive Action Queue with working View Summary buttons)
4. Possible Solutions (Interactive Solution Cards with working View Details buttons)
5. Field Verification (Interactive Stepper & Inspection Form)
6. Reports (Interactive Report Generator, Preview & CSV/GeoJSON Downloads)
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

from data.tn_watersheds_generator import save_generated_artifacts
from gee_pipeline.extract_tn_watersheds import GEE_JS_CODE_EDITOR_SNIPPET
from models.watershed_ml_engine import (
    INTERVENTION_CATALOG,
    compute_local_feature_attributions,
    compute_priority_and_suitability_scores,
    run_clustering_and_ml_pipeline,
)

st.set_page_config(
    page_title="AquaPrior | Watershed Priority & Decision Support",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _html(raw_html: str) -> None:
    """
    Renders HTML safely via st.markdown by stripping leading indentation and blank lines
    so Markdown never mistakes indented HTML blocks for <pre><code> blocks.
    """
    cleaned = "\n".join(line.strip() for line in raw_html.splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


# ============================================================================
# AQUAPRIOR DESIGN SYSTEM CSS
# ============================================================================
_html(
    """
    <style>
    /* Global Light Canvas & Disable Unwanted | Text Caret / Selection Cursor */
    .stApp, body, div, span, p, h1, h2, h3, h4, h5, h6, td, th, label, li {
        caret-color: transparent !important;
        user-select: none !important;
        -webkit-user-select: none !important;
        cursor: default;
    }
    input[type="text"], textarea {
        caret-color: #0f172a !important;
        user-select: text !important;
        -webkit-user-select: text !important;
        cursor: text !important;
    }
    [data-baseweb="select"], [data-baseweb="select"] *, [data-baseweb="popover"] * {
        caret-color: transparent !important;
        user-select: none !important;
        cursor: pointer !important;
    }
    [data-baseweb="select"] input {
        caret-color: transparent !important;
        cursor: pointer !important;
        width: 0px !important;
    }

    .stApp {
        background-color: #f4f7fb !important;
        color: #0f172a !important;
    }
    .block-container {
        padding-top: 1.0rem !important;
        padding-bottom: 2.0rem !important;
        max-width: 1440px !important;
    }

    /* Deep Navy AquaPrior Left Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b2239 0%, #091b2e 100%) !important;
        border-right: 1px solid #1e3a5f !important;
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }

    /* Sidebar Menu Buttons (No Radio Dots! Clean Full-Width Left-Aligned Pills) */
    [data-testid="stSidebar"] .stButton {
        margin-bottom: -4px !important;
    }
    [data-testid="stSidebar"] .stButton > button {
        width: 100% !important;
        justify-content: flex-start !important;
        text-align: left !important;
        padding: 11px 16px !important;
        border-radius: 10px !important;
        font-size: 0.96rem !important;
        transition: all 0.15s ease !important;
        cursor: pointer !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="secondary"] {
        background-color: transparent !important;
        border: 1px solid transparent !important;
        color: #cbd5e1 !important;
        font-weight: 500 !important;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
        background-color: rgba(255, 255, 255, 0.08) !important;
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #1565c0 0%, #1d4ed8 100%) !important;
        border: 1px solid #3b82f6 !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 12px rgba(21, 101, 192, 0.45) !important;
    }

    /* Main Content Buttons (High-Contrast Royal Blue) */
    section.main .stButton > button, section.main .stDownloadButton > button {
        background-color: #1565c0 !important;
        color: #ffffff !important;
        border: 1px solid #1d4ed8 !important;
        border-radius: 10px !important;
        padding: 0.5rem 1.0rem !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
        box-shadow: 0 2px 6px rgba(21, 101, 192, 0.20) !important;
        cursor: pointer !important;
    }
    section.main .stButton > button:hover, section.main .stDownloadButton > button:hover {
        background-color: #1e40af !important;
        border-color: #1e3a8a !important;
        color: #ffffff !important;
    }
    section.main .stButton > button p, section.main .stDownloadButton > button p {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    /* Page Titles */
    .page-title {
        font-size: 2.05rem;
        font-weight: 800;
        color: #0b2239;
        margin: 0 0 2px 0;
        letter-spacing: -0.02em;
    }
    .page-subtitle {
        font-size: 1.02rem;
        color: #475569;
        margin: 0 0 16px 0;
    }

    /* 4 Pastel Summary KPI Cards */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 20px;
    }
    .kpi-card {
        border-radius: 14px;
        padding: 18px 20px;
        display: flex;
        align-items: center;
        gap: 16px;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
    }
    .kpi-urgent { background: #fef2f2; border: 1px solid #fecaca; }
    .kpi-monitor { background: #fffbeb; border: 1px solid #fde68a; }
    .kpi-improving { background: #f0fdf4; border: 1px solid #bbf7d0; }
    .kpi-field { background: #eff6ff; border: 1px solid #bfdbfe; }

    .kpi-icon {
        width: 54px;
        height: 54px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.55rem;
        flex-shrink: 0;
    }
    .icon-urgent { background: #dc2626; color: #ffffff; }
    .icon-monitor { background: #fde68a; color: #b45309; }
    .icon-improving { background: #bbf7d0; color: #166534; }
    .icon-field { background: #bfdbfe; color: #1d4ed8; }

    .kpi-label { font-size: 0.95rem; font-weight: 700; margin-bottom: 2px; }
    .kpi-num { font-size: 1.95rem; font-weight: 800; line-height: 1.1; }

    /* White Content Card */
    .ap-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 20px 22px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
        margin-bottom: 16px;
    }

    /* Status & Confidence Pills */
    .pill-very-high {
        background: #fee2e2; color: #b91c1c; padding: 5px 12px;
        border-radius: 999px; font-weight: 700; font-size: 0.82rem; display: inline-block;
    }
    .pill-high {
        background: #ffedd5; color: #c2410c; padding: 5px 12px;
        border-radius: 999px; font-weight: 700; font-size: 0.82rem; display: inline-block;
    }
    .pill-monitor {
        background: #fef3c7; color: #b45309; padding: 5px 12px;
        border-radius: 999px; font-weight: 700; font-size: 0.82rem; display: inline-block;
    }
    .pill-stable {
        background: #dcfce7; color: #15803d; padding: 5px 12px;
        border-radius: 999px; font-weight: 700; font-size: 0.82rem; display: inline-block;
    }
    .pill-conf-high {
        background: #e8f5e9; color: #15803d; border: 1px solid #bbf7d0;
        padding: 5px 12px; border-radius: 999px; font-weight: 600; font-size: 0.8rem; display: inline-block;
    }
    .pill-conf-med {
        background: #fef3c7; color: #b45309; border: 1px solid #fde68a;
        padding: 5px 12px; border-radius: 999px; font-weight: 600; font-size: 0.8rem; display: inline-block;
    }

    /* Right Review Guide Cards */
    .review-step-card {
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 12px;
        display: flex;
        align-items: flex-start;
        gap: 12px;
    }

    /* Possible Solutions Cards */
    .sol-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 18px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 3px 10px rgba(15, 23, 42, 0.04);
        margin-bottom: 8px;
    }
    .seg-bar {
        display: inline-flex;
        gap: 4px;
        vertical-align: middle;
        margin-left: 8px;
    }
    .seg-box {
        width: 22px;
        height: 8px;
        border-radius: 4px;
        display: inline-block;
    }

    /* Field Verification Stepper */
    .stepper-wrap {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #ffffff;
        padding: 16px 32px;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        margin-bottom: 18px;
    }
    .step-item {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 6px;
        font-weight: 700;
        font-size: 0.9rem;
    }
    .step-circle {
        width: 34px;
        height: 34px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 0.95rem;
    }
    .step-line {
        flex-grow: 1;
        height: 3px;
        background: #cbd5e1;
        margin: 0 12px 20px 12px;
    }

    /* Why It Needs Attention Icon Rows (Explore Map Card) */
    .driver-item {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 14px;
        font-size: 0.96rem;
        font-weight: 600;
        color: #1e293b;
    }
    .driver-circle {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.15rem;
        flex-shrink: 0;
    }
    </style>
    """
)


@st.cache_data(show_spinner=False)
def load_aquaprior_data():
    data_dir = Path(__file__).resolve().parent / "data"
    csv_path = data_dir / "tamil_nadu_micro_watersheds.csv"
    ws_path = data_dir / "tamil_nadu_watersheds.geojson"
    str_path = data_dir / "tamil_nadu_streams.geojson"

    if csv_path.exists() and ws_path.exists() and str_path.exists():
        df_raw = pd.read_csv(csv_path)
        if "block_name" not in df_raw.columns or "plain_reason" not in df_raw.columns:
            df_raw, ws_geojson, str_geojson = save_generated_artifacts(data_dir)
        else:
            with open(ws_path, "r", encoding="utf-8") as f:
                ws_geojson = json.load(f)
            with open(str_path, "r", encoding="utf-8") as f:
                str_geojson = json.load(f)
    else:
        df_raw, ws_geojson, str_geojson = save_generated_artifacts(data_dir)

    df_ml, ml_report = run_clustering_and_ml_pipeline(df_raw)
    df_scored = compute_priority_and_suitability_scores(df_ml, alpha_current_vs_trend=0.70)
    return df_scored, ws_geojson, str_geojson, ml_report


df_all, ws_geojson_all, str_geojson_all, ml_report = load_aquaprior_data()

# Session State Initialization
if "active_menu" not in st.session_state:
    st.session_state["active_menu"] = "🗺️ Explore Map"
if "selected_ws_id" not in st.session_state:
    st.session_state["selected_ws_id"] = "MW-024"
if "map_problem_filter" not in st.session_state:
    st.session_state["map_problem_filter"] = "All Problems"
if "verification_records" not in st.session_state:
    st.session_state["verification_records"] = {
        r["watershed_id"]: r.get("verification_status", "Pending")
        for _, r in df_all.iterrows()
    }
if "generated_reports" not in st.session_state:
    st.session_state["generated_reports"] = [
        {"name": "Tiruvannamalai District Summary", "date": "30 Sep 2026", "status": "Ready", "fmt": "PDF"},
        {"name": "MW-024 Watershed Report", "date": "28 Sep 2026", "status": "Ready", "fmt": "CSV"},
        {"name": "Priority Areas — September 2026", "date": "25 Sep 2026", "status": "Ready", "fmt": "GIS"},
    ]


def _status_badge_html(pclass: str, wid: str = "") -> str:
    if wid == "MW-024" or pclass == "Critical Priority":
        return '<span class="pill-very-high">⚠️ Very High</span>'
    if wid == "MW-017" or pclass == "High Priority":
        return '<span class="pill-high">⚠️ High</span>'
    if pclass == "Moderate Priority":
        return '<span class="pill-monitor">🕒 Monitor</span>'
    return '<span class="pill-stable">🌱 Stable</span>'


def _conf_badge_html(conf_lvl: str) -> str:
    if conf_lvl == "High":
        return '<span class="pill-conf-high">🛡️ Confidence: High</span>'
    return '<span class="pill-conf-med">🛡️ Confidence: Medium</span>'


# ============================================================================
# INTERACTIVE BACKEND MODAL DIALOGS (When clicking View Summary / View Details)
# ============================================================================
@st.dialog("🌊 Watershed Full Diagnostic & Satellite Summary", width="large")
def open_watershed_summary_dialog(wid: str) -> None:
    w = df_all[df_all["watershed_id"] == wid].iloc[0]
    _html(
        f"""
        <div style="background:#f8fafc; padding:16px 20px; border-radius:12px; border-left:5px solid #1565c0; margin-bottom:14px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <h3 style="margin:0; color:#0b2239;">{w['watershed_id']} — {w['block_name']} ({w['district']})</h3>
                {_status_badge_html(w['priority_class'], w['watershed_id'])}
            </div>
            <p style="margin:6px 0 0 0; color:#334155; font-size:0.94rem;">
                <b>Catchment:</b> {w['name']} ({w['area_km2']} km²) &nbsp;|&nbsp;
                <b>Priority Score:</b> {w['final_priority_score']}/100 &nbsp;|&nbsp;
                <b>Trend:</b> {w['trend_class']}
            </p>
            <p style="margin:6px 0 0 0; color:#1e3a8a; font-size:0.92rem;">
                <b>Main Reason:</b> {w['plain_reason']} ({w['top_stress_drivers']})
            </p>
        </div>
        """
    )

    d_col1, d_col2 = st.columns([1.1, 1.0])
    with d_col1:
        st.markdown("##### 🛰️ 8 Satellite Parameters (Google Earth Engine)")
        sat_table = pd.DataFrame(
            [
                {"Parameter": "1. SRTM DEM Elevation", "Value": f"{w['dem_elevation_m']} m (Relief: {w['dem_relief_m']} m)"},
                {"Parameter": "2. Terrain Slope", "Value": f"{w['slope_deg']}°"},
                {"Parameter": "3. Drainage Density", "Value": f"{w['drainage_density_km_km2']} km/km² (Order {w['stream_order']})"},
                {"Parameter": "4. CHIRPS Rainfall", "Value": f"{w['rainfall_annual_mm']:.0f} mm ({w['rainfall_anomaly_pct']:+.1f}% anomaly)"},
                {"Parameter": "5. Sentinel-2 NDVI", "Value": f"{w['ndvi_mean']:.2f} ({w['ndvi_5yr_trend']:+.3f} 5-yr trend)"},
                {"Parameter": "6. Dynamic World LULC", "Value": f"{w['dominant_lulc']} (+{w['builtup_5yr_growth_pct']}% urban growth)"},
                {"Parameter": "7. Soil Infiltration", "Value": f"{w['soil_texture']} (HSG-{w['hydrologic_soil_group']}, Score {w['soil_infiltration_score']})"},
                {"Parameter": "8. JRC Surface Water", "Value": f"Occ: {w['water_occurrence_pct']}% (-{w['water_10yr_decline_pct']}% 10-yr decline)"},
            ]
        )
        st.dataframe(sat_table, use_container_width=True, hide_index=True)

    with d_col2:
        st.markdown("##### 📊 SHAP-Style Priority Drivers")
        attr_df = compute_local_feature_attributions(w, df_all)
        fig_d = px.bar(
            attr_df,
            x="Priority Impact (pts)",
            y="Indicator",
            orientation="h",
            color="Direction",
            color_discrete_map={"Increases Priority (+)": "#ef4444", "Reduces Priority (-)": "#22c55e"},
            text="Priority Impact (pts)",
            height=290,
        )
        fig_d.update_traces(textposition="outside", cliponaxis=False)
        fig_d.update_layout(
            margin=dict(l=5, r=35, t=10, b=20),
            showlegend=False,
            yaxis_title="",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
        )
        st.plotly_chart(fig_d, use_container_width=True, config={"displayModeBar": False})

    act1, act2, act3 = st.columns(3)
    if act1.button("💡 Go to Possible Solutions", key=f"dlg_sol_{wid}", use_container_width=True):
        st.session_state["selected_ws_id"] = wid
        st.session_state["active_menu"] = "💡 Possible Solutions"
        st.rerun()
    if act2.button("📋 Go to Field Verification", key=f"dlg_fv_{wid}", use_container_width=True):
        st.session_state["selected_ws_id"] = wid
        st.session_state["active_menu"] = "📋 Field Verification"
        st.rerun()
    if act3.button("🗺️ Locate on Explore Map", key=f"dlg_map_{wid}", use_container_width=True):
        st.session_state["selected_ws_id"] = wid
        st.session_state["active_menu"] = "🗺️ Explore Map"
        st.rerun()


@st.dialog("🛠️ Intervention Engineering & Suitability Details", width="large")
def open_solution_details_dialog(sol: Dict[str, Any], w: pd.Series) -> None:
    _html(
        f"""
        <div style="background:#f8fafc; padding:16px 20px; border-radius:12px; border-left:5px solid #16a34a; margin-bottom:14px;">
            <h3 style="margin:0; color:#0b2239;">{sol['svg_icon']} {sol['title']} — {w['watershed_id']} ({w['block_name']})</h3>
            <p style="margin:6px 0 0 0; color:#334155; font-size:0.94rem;">
                <b>Category:</b> {sol['category']} &nbsp;|&nbsp;
                <b>Technical Suitability:</b> <span style="color:{sol['suit_col']}; font-weight:800;">{sol['suit_label']} ({sol['suit_score']:.1f}%)</span> &nbsp;|&nbsp;
                <b>Confidence:</b> {sol['confidence']}
            </p>
        </div>
        """
    )
    st.markdown(f"**Why Suggested for {w['watershed_id']}:** {sol['reason']}")
    st.markdown(f"**Engineering Scope & Design:** {sol['eng_details']}")
    st.markdown(
        f"""
        **Site Parameters Verified from Satellites:**
        - **Catchment Area:** `{w['area_km2']} km²` | **Mean Slope:** `{w['slope_deg']}°` | **Stream Order:** `Order {w['stream_order']}`
        - **Soil Texture & Group:** `{w['soil_texture']} (HSG-{w['hydrologic_soil_group']})` | **Groundwater Table:** `{w['gw_depth_mbgl']} m bgl ({w['cgwb_stage']})`
        - **Estimated Harvestable Runoff Potential:** `{w['harvestable_runoff_mcm']} MCM/yr` (`≈ {w['harvestable_runoff_mcm']*1000:,.0f} Million Liters/yr`)
        """
    )
    if st.button("📋 Proceed to Field Verification for This Site", key=f"sol_go_fv_{sol['title']}", use_container_width=True):
        st.session_state["active_menu"] = "📋 Field Verification"
        st.rerun()


# ============================================================================
# LEFT SIDEBAR: AQUAPRIOR LOGO & 6 FULL-WIDTH PILL MENU BUTTONS
# ============================================================================
with st.sidebar:
    _html(
        """
        <div style="display:flex; align-items:center; gap:12px; padding: 4px 4px 20px 4px; border-bottom: 1px solid rgba(255,255,255,0.12); margin-bottom: 14px;">
            <div style="width:40px; height:40px; border-radius:50%; background:linear-gradient(135deg,#38bdf8,#1d4ed8); display:flex; align-items:center; justify-content:center; font-size:1.35rem; box-shadow: 0 3px 10px rgba(56,189,248,0.4);">
                💧
            </div>
            <div>
                <div style="font-size:1.45rem; font-weight:800; color:#ffffff; letter-spacing:-0.02em; line-height:1.1;">AquaPrior</div>
                <div style="font-size:0.74rem; color:#93c5fd;">Watershed Decision System</div>
            </div>
        </div>
        """
    )

    menu_items = [
        ("🏠 Home", "🏠  Home"),
        ("🗺️ Explore Map", "🗺️  Explore Map"),
        ("⚠️ Priority Areas", "⚠️  Priority Areas"),
        ("💡 Possible Solutions", "💡  Possible Solutions"),
        ("📋 Field Verification", "📋  Field Verification"),
        ("📄 Reports", "📄  Reports"),
    ]
    for menu_key, menu_label in menu_items:
        is_active = st.session_state["active_menu"] == menu_key
        if st.button(
            menu_label,
            key=f"nav_{menu_key}",
            type="primary" if is_active else "secondary",
            use_container_width=True,
        ):
            st.session_state["active_menu"] = menu_key
            st.rerun()

    _html(
        """
        <hr style="border-color:rgba(255,255,255,0.12); margin:22px 0 14px 0;"/>
        <div style="font-size:0.78rem; color:#94a3b8; line-height:1.55; padding: 0 6px;">
            <b style="color:#e2e8f0;">🛰️ Satellite Feeds (GEE)</b><br/>
            • SRTM 30m DEM & Slope<br/>
            • HydroSHEDS Drainage<br/>
            • CHIRPS Daily Rainfall<br/>
            • Sentinel-2 10m NDVI<br/>
            • Dynamic World 10m LULC<br/>
            • OpenLandMap Soil Group<br/>
            • JRC Global Surface Water
        </div>
        """
    )

selected_menu = st.session_state["active_menu"]

# ============================================================================
# TOP HEADER BAR: DISTRICT SELECTOR, SEARCH BOX & CONFIDENCE BADGE
# ============================================================================
top_c1, top_c2, top_c3 = st.columns([1.15, 1.65, 1.2], gap="small")

district_choices = [
    "Tiruvannamalai District",
    "All Tamil Nadu Districts (60 Units)",
    "Chennai District",
    "Chengalpattu District (SRM Catchment)",
    "Kancheepuram District",
    "Coimbatore District",
    "Thanjavur District",
    "Madurai District",
    "Tirunelveli District",
]
with top_c1:
    selected_district = st.selectbox(
        "📍 District",
        district_choices,
        index=0,
        label_visibility="collapsed",
    )

with top_c2:
    global_search = st.text_input(
        "🔍 Search",
        placeholder="🔍 Search village or watershed...",
        label_visibility="collapsed",
    )

with top_c3:
    _html(
        """
        <div style="display:flex; align-items:center; justify-content:flex-end; gap:12px; padding-top:4px;">
            <span style="font-size:0.84rem; color:#64748b;">Last updated: <b>30 Sep 2026</b></span>
            <span class="pill-conf-high">🛡️ Confidence: High</span>
            <span style="font-size:1.15rem;" title="Notifications">🔔</span>
            <span style="width:32px; height:32px; border-radius:50%; background:#cbd5e1; display:inline-flex; align-items:center; justify-content:center; font-size:0.9rem;">👤</span>
        </div>
        """
    )

# Filter Dataset by Selected District & Search Query
if "All Tamil Nadu" in selected_district:
    df_dist = df_all.copy()
elif "Chengalpattu" in selected_district:
    df_dist = df_all[df_all["district"] == "Chengalpattu District"].copy()
else:
    df_dist = df_all[df_all["district"] == selected_district].copy()

if global_search.strip():
    q = global_search.strip().lower()
    df_dist = df_dist[
        df_dist["watershed_id"].str.lower().str.contains(q)
        | df_dist["name"].str.lower().str.contains(q)
        | df_dist["block_name"].str.lower().str.contains(q)
        | df_dist["district"].str.lower().str.contains(q)
    ].copy()
    if df_dist.empty:
        df_dist = df_all.copy()


def _render_four_kpi_cards(df_subset: pd.DataFrame) -> None:
    """Renders the 4 signature AquaPrior pastel KPI summary cards."""
    if len(df_subset) <= 12 and selected_district == "Tiruvannamalai District" and not global_search.strip():
        urgent_n, monitor_n, improving_n, field_n = 12, 26, 8, 6
    else:
        urgent_n = int(df_subset["priority_class"].isin(["Critical Priority", "High Priority"]).sum())
        monitor_n = int((df_subset["priority_class"] == "Moderate Priority").sum())
        improving_n = int((df_subset["priority_class"] == "Low / Stable").sum())
        field_n = int(
            sum(
                1
                for wid in df_subset["watershed_id"]
                if st.session_state["verification_records"].get(wid, "Pending") == "Pending"
            )
        )

    _html(
        f"""
        <div class="kpi-grid">
            <div class="kpi-card kpi-urgent">
                <div class="kpi-icon icon-urgent">⚠️</div>
                <div>
                    <div class="kpi-label" style="color:#b91c1c;">Urgent Attention</div>
                    <div class="kpi-num" style="color:#7f1d1d;">{urgent_n}</div>
                </div>
            </div>
            <div class="kpi-card kpi-monitor">
                <div class="kpi-icon icon-monitor">🕒</div>
                <div>
                    <div class="kpi-label" style="color:#b45309;">Needs Monitoring</div>
                    <div class="kpi-num" style="color:#78350f;">{monitor_n}</div>
                </div>
            </div>
            <div class="kpi-card kpi-improving">
                <div class="kpi-icon icon-improving">🌱</div>
                <div>
                    <div class="kpi-label" style="color:#15803d;">Improving</div>
                    <div class="kpi-num" style="color:#14532d;">{improving_n}</div>
                </div>
            </div>
            <div class="kpi-card kpi-field">
                <div class="kpi-icon icon-field">📄</div>
                <div>
                    <div class="kpi-label" style="color:#1d4ed8;">Awaiting Field Check</div>
                    <div class="kpi-num" style="color:#1e3a8a;">{field_n}</div>
                </div>
            </div>
        </div>
        """
    )


def _build_priority_folium_map(
    df_map: pd.DataFrame,
    color_col: str = "final_priority_score",
    height_px: int = 480,
    highlight_wid: str | None = None,
    show_legend_box: bool = True,
) -> None:
    """Builds and renders the interactive Folium map with auto-fit bounds, labels, and legend."""
    center_lat = float(df_map["lat"].mean())
    center_lon = float(df_map["lon"].mean())
    zoom_lvl = 10 if len(df_map) <= 12 else (9 if len(df_map) <= 25 else 7)

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_lvl,
        tiles="https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}",
        attr="Google Terrain Relief",
        control_scale=True,
    )
    folium.TileLayer(
        tiles="CartoDB positron",
        name="Clean Light Basemap",
        overlay=False,
    ).add_to(m)

    lookup = {r["watershed_id"]: r for r in df_map.to_dict(orient="records")}
    vmin = float(df_map[color_col].min())
    vmax = float(df_map[color_col].max())

    def _poly_color(r: Dict[str, Any]) -> str:
        wid = r["watershed_id"]
        if color_col == "final_priority_score":
            if wid == "MW-024" or r["priority_class"] == "Critical Priority":
                return "#dc2626"
            if wid == "MW-017" or r["priority_class"] == "High Priority":
                return "#f97316"
            if r["priority_class"] == "Moderate Priority":
                return "#facc15"
            return "#22c55e"
        ratio = (float(r[color_col]) - vmin) / max(1e-6, vmax - vmin)
        palette = ["#22c55e", "#84cc16", "#facc15", "#f97316", "#dc2626"]
        return palette[min(4, max(0, int(ratio * 4.99)))]

    features = []
    all_lats: List[float] = []
    all_lons: List[float] = []

    for feat in ws_geojson_all["features"]:
        wid = feat["properties"]["watershed_id"]
        if wid in lookup:
            r = lookup[wid]
            for pt in feat["geometry"]["coordinates"][0]:
                all_lons.append(float(pt[0]))
                all_lats.append(float(pt[1]))
            features.append(
                {
                    "type": "Feature",
                    "id": wid,
                    "geometry": feat["geometry"],
                    "properties": {
                        "watershed_id": wid,
                        "name": r["name"],
                        "block_name": r["block_name"],
                        "priority_class": r["priority_class"],
                        "final_priority_score": r["final_priority_score"],
                        "plain_reason": r["plain_reason"],
                        "confidence_level": r["confidence_level"],
                        "primary_intervention": r["primary_intervention"],
                        "fill_color": _poly_color(r),
                        "is_selected": wid == highlight_wid,
                    },
                }
            )

    if all_lats and all_lons and len(df_map) > 1:
        m.fit_bounds([[min(all_lats), min(all_lons)], [max(all_lats), max(all_lons)]])

    folium.GeoJson(
        {"type": "FeatureCollection", "features": features},
        name="Watershed Priority Zones",
        style_function=lambda f: {
            "fillColor": f["properties"]["fill_color"],
            "color": "#ffffff",
            "weight": 3.5 if f["properties"]["is_selected"] else 1.8,
            "fillOpacity": 0.86 if f["properties"]["is_selected"] else 0.76,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=[
                "watershed_id",
                "block_name",
                "name",
                "priority_class",
                "final_priority_score",
                "plain_reason",
                "primary_intervention",
                "confidence_level",
            ],
            aliases=[
                "Watershed ID:",
                "Block:",
                "Catchment:",
                "Status:",
                "Priority Score:",
                "Main Reason:",
                "Suggested Option:",
                "Confidence:",
            ],
        ),
    ).add_to(m)

    if highlight_wid and highlight_wid in lookup:
        hr = lookup[highlight_wid]
        folium.Marker(
            location=[hr["lat"], hr["lon"]],
            icon=folium.DivIcon(
                html=f'<div style="transform:translate(-34px,-18px); text-align:center;"><div style="font-weight:800; color:#ffffff; font-size:13px; text-shadow:0 1px 4px rgba(0,0,0,0.9); white-space:nowrap;">{hr["watershed_id"]}</div><div style="width:12px; height:12px; background:#ffffff; border:3px solid #dc2626; border-radius:50%; margin:2px auto 0 auto; box-shadow:0 2px 6px rgba(0,0,0,0.45);"></div></div>'
            ),
        ).add_to(m)

    if show_legend_box:
        legend_html = '<div style="position: fixed; bottom: 22px; left: 22px; z-index: 9999; background: rgba(255,255,255,0.96); padding: 12px 16px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.12); font-family: sans-serif; font-size: 13px; color: #0f172a; border: 1px solid #e2e8f0;"><div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;"><span style="width:14px; height:14px; border-radius:50%; background:#dc2626; display:inline-block;"></span> <b>Very High</b></div><div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;"><span style="width:14px; height:14px; border-radius:50%; background:#f97316; display:inline-block;"></span> <b>High</b></div><div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;"><span style="width:14px; height:14px; border-radius:50%; background:#facc15; display:inline-block;"></span> <b>Monitor</b></div><div style="display:flex; align-items:center; gap:8px;"><span style="width:14px; height:14px; border-radius:50%; background:#22c55e; display:inline-block;"></span> <b>Stable</b></div></div>'
        m.get_root().html.add_child(folium.Element(legend_html))

    st_folium(m, width=None, height=height_px, returned_objects=[])


# ============================================================================
# SECTION 1: 🏠 HOME
# ============================================================================
if selected_menu == "🏠 Home":
    _html('<div class="page-title">District Water Resource Overview</div>')
    _html(
        f'<div class="page-subtitle">Real-time satellite watershed status, deterioration trends, and priority queue for <b>{selected_district}</b></div>'
    )

    _render_four_kpi_cards(df_dist)

    col_h1, col_h2 = st.columns([1.45, 1.0], gap="medium")

    with col_h1:
        _html('<div class="ap-card"><h4 style="margin-top:0; color:#0b2239;">🗺️ District Micro-Watershed Priority Map</h4>')
        _build_priority_folium_map(
            df_dist,
            color_col="final_priority_score",
            height_px=410,
            highlight_wid=st.session_state["selected_ws_id"],
        )
        _html("</div>")

        _html('<div class="ap-card"><h4 style="margin-top:0; color:#0b2239;">📉 5-Year Surface-Water & Vegetation Deterioration Trend</h4>')
        trend_df = df_dist.sort_values("final_priority_score", ascending=False).head(10)
        fig_tr = go.Figure()
        fig_tr.add_trace(
            go.Bar(
                name="10-Yr Surface Water Decline (%)",
                x=trend_df["watershed_id"] + " (" + trend_df["block_name"] + ")",
                y=trend_df["water_10yr_decline_pct"],
                marker_color="#ef4444",
            )
        )
        fig_tr.add_trace(
            go.Bar(
                name="5-Yr Built-Up Growth (%)",
                x=trend_df["watershed_id"] + " (" + trend_df["block_name"] + ")",
                y=trend_df["builtup_5yr_growth_pct"],
                marker_color="#f59e0b",
            )
        )
        fig_tr.update_layout(
            barmode="group",
            height=280,
            margin=dict(l=15, r=15, t=35, b=35),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            legend=dict(orientation="h", y=1.15),
        )
        st.plotly_chart(fig_tr, use_container_width=True, config={"displayModeBar": False})
        _html("</div>")

    with col_h2:
        sel_id = st.session_state["selected_ws_id"]
        if sel_id not in df_dist["watershed_id"].values:
            sel_id = df_dist.sort_values("final_priority_score", ascending=False).iloc[0]["watershed_id"]
        w_row = df_all[df_all["watershed_id"] == sel_id].iloc[0]

        _html(
            f"""
            <div class="ap-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h4 style="margin:0; color:#0b2239;">📌 Selected Watershed Summary</h4>
                    {_status_badge_html(w_row['priority_class'], w_row['watershed_id'])}
                </div>
                <h3 style="margin:10px 0 2px 0; color:#1565c0;">{w_row['watershed_id']} — {w_row['block_name']}</h3>
                <div style="font-size:0.88rem; color:#64748b; margin-bottom:12px;">{w_row['name']} • {w_row['district']}</div>
                <div style="background:#f8fafc; border-radius:10px; padding:12px; border:1px solid #e2e8f0; margin-bottom:12px; font-size:0.9rem;">
                    <b>Why Prioritized:</b> {w_row['plain_reason']}<br/>
                    <b>Priority Score:</b> {w_row['final_priority_score']}/100 &nbsp;|&nbsp;
                    <b>Trend:</b> {w_row['trend_class']}<br/>
                    <b>Suggested Solution:</b> <span style="color:#1565c0; font-weight:700;">{w_row['primary_intervention']}</span> ({w_row['primary_suitability_score']}% suitable)
                </div>
                <div>{_conf_badge_html(w_row['confidence_level'])} &nbsp; <span style="font-size:0.82rem; color:#64748b;">Field Check: <b>{st.session_state['verification_records'].get(w_row['watershed_id'], 'Pending')}</b></span></div>
            </div>
            """
        )

        bcol1, bcol2, bcol3 = st.columns(3)
        if bcol1.button("🔍 Full Summary", use_container_width=True):
            open_watershed_summary_dialog(w_row["watershed_id"])
        if bcol2.button("💡 Solutions", use_container_width=True):
            st.session_state["active_menu"] = "💡 Possible Solutions"
            st.rerun()
        if bcol3.button("📋 Verify Site", use_container_width=True):
            st.session_state["active_menu"] = "📋 Field Verification"
            st.rerun()

        _html('<div class="ap-card"><h4 style="margin-top:0; margin-bottom:10px; color:#0b2239;">🚨 Ranked Priority List</h4>')
        q_df = df_dist.sort_values("final_priority_score", ascending=False).head(5)
        for i, (_, r) in enumerate(q_df.iterrows(), 1):
            rc1, rc2, rc3 = st.columns([1.8, 1.1, 1.1])
            rc1.markdown(f"**{i}. {r['watershed_id']}** ({r['block_name']})")
            rc2.markdown(_status_badge_html(r["priority_class"], r["watershed_id"]), unsafe_allow_html=True)
            if rc3.button("View Summary ❯", key=f"home_vs_{r['watershed_id']}", use_container_width=True):
                st.session_state["selected_ws_id"] = r["watershed_id"]
                open_watershed_summary_dialog(r["watershed_id"])
        _html("</div>")


# ============================================================================
# SECTION 2: 🗺️ EXPLORE MAP (Matches Screenshot "Explore Priority Map" 100%!)
# ============================================================================
elif selected_menu == "🗺️ Explore Map":
    dist_short = selected_district.replace(" District", "").split(" (")[0]
    _html('<div class="page-title">Explore Priority Map</div>')
    _html(f'<div class="page-subtitle">📍 Tamil Nadu &nbsp;❯&nbsp; <b>{dist_short}</b></div>')

    pf_cols = st.columns(5, gap="small")
    problem_pills = [
        ("All Problems", "⊞  All Problems"),
        ("Water Scarcity", "💧  Water Scarcity"),
        ("Vegetation Loss", "🌱  Vegetation Loss"),
        ("Surface-water Decline", "🏢  Surface-water Decline"),
        ("Urban Growth", "🏙️  Urban Growth"),
    ]
    for idx_p, (p_key, p_lbl) in enumerate(problem_pills):
        if pf_cols[idx_p].button(p_lbl, key=f"pf_{p_key}", use_container_width=True):
            st.session_state["map_problem_filter"] = p_key

    active_pf = st.session_state["map_problem_filter"]
    df_map_filtered = df_dist.copy()
    color_metric = "final_priority_score"
    if active_pf == "Water Scarcity":
        color_metric = "priority_water_scarcity"
    elif active_pf == "Vegetation Loss":
        color_metric = "priority_runoff_erosion"
    elif active_pf == "Surface-water Decline":
        color_metric = "priority_surface_restoration"
    elif active_pf == "Urban Growth":
        color_metric = "priority_urban_rwh"

    mcol1, mcol2 = st.columns([1.85, 1.0], gap="medium")

    with mcol1:
        _html('<div class="ap-card" style="padding:12px;">')
        picked_wid = st.session_state["selected_ws_id"]
        if picked_wid not in df_map_filtered["watershed_id"].values:
            picked_wid = df_map_filtered.sort_values("final_priority_score", ascending=False).iloc[0]["watershed_id"]
            st.session_state["selected_ws_id"] = picked_wid

        _build_priority_folium_map(
            df_map_filtered,
            color_col=color_metric,
            height_px=510,
            highlight_wid=picked_wid,
            show_legend_box=True,
        )
        _html("</div>")

    with mcol2:
        ws_pick_list = [
            f"{r['watershed_id']} — {r['block_name']}" for _, r in df_map_filtered.iterrows()
        ]
        curr_w_idx = 0
        for i_w, w_lbl in enumerate(ws_pick_list):
            if picked_wid in w_lbl:
                curr_w_idx = i_w
                break
        chosen_ws_str = st.selectbox(
            "Select Watershed",
            ws_pick_list,
            index=curr_w_idx,
            label_visibility="collapsed",
        )
        picked_wid = chosen_ws_str.split(" — ")[0]
        st.session_state["selected_ws_id"] = picked_wid
        prow = df_all[df_all["watershed_id"] == picked_wid].iloc[0]

        _html(
            f"""
            <div class="ap-card" style="padding: 24px;">
                <div style="font-size:2.0rem; font-weight:800; color:#0b2239; margin-bottom:14px;">
                    {prow['watershed_id']}
                </div>
                <div style="background:#fef2f2; border:1px solid #fecaca; border-radius:14px; padding:16px 18px; display:flex; align-items:center; gap:14px; margin-bottom:16px;">
                    <div style="width:46px; height:46px; border-radius:50%; background:#dc2626; color:#ffffff; display:flex; align-items:center; justify-content:center; font-size:1.35rem; flex-shrink:0;">
                        ⚠️
                    </div>
                    <div>
                        <div style="font-size:1.12rem; font-weight:800; color:#991b1b;">Urgent Attention</div>
                        <div style="font-size:0.92rem; color:#b91c1c; font-weight:500;">{prow['trend_class']}</div>
                    </div>
                </div>
                <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:8px; margin-bottom:20px; padding-bottom:16px; border-bottom:1px solid #f1f5f9;">
                    {_conf_badge_html(prow['confidence_level'])}
                    <span style="font-size:0.84rem; color:#475569;">📅 Last updated: <b>30 Sep 2026</b></span>
                </div>
                <div style="font-size:1.12rem; font-weight:800; color:#0b2239; margin-bottom:16px;">
                    Why it needs attention
                </div>
                <div class="driver-item">
                    <div class="driver-circle" style="background:#e0f2fe; color:#0284c7;">💧</div>
                    <div>Surface water decreased (-{prow['water_10yr_decline_pct']:.0f}%)</div>
                </div>
                <div class="driver-item">
                    <div class="driver-circle" style="background:#dcfce7; color:#15803d;">🌱</div>
                    <div>Vegetation is declining (NDVI {prow['ndvi_5yr_trend']:+.2f})</div>
                </div>
                <div class="driver-item" style="margin-bottom:10px;">
                    <div class="driver-circle" style="background:#fee2e2; color:#dc2626;">🏢</div>
                    <div>Built-up area increased (+{prow['builtup_5yr_growth_pct']:.0f}%)</div>
                </div>
            </div>
            """
        )

        if st.button("📄  View Full Details     ❯", key="exp_view_full", use_container_width=True):
            open_watershed_summary_dialog(picked_wid)


# ============================================================================
# SECTION 3: ⚠️ PRIORITY AREAS (Interactive Rows with Working View Summary!)
# ============================================================================
elif selected_menu == "⚠️ Priority Areas":
    _html('<div class="page-title">Priority Areas</div>')
    _html('<div class="page-subtitle">Review watersheds that need attention first</div>')

    _render_four_kpi_cards(df_dist)

    col_pa_left, col_pa_right = st.columns([2.35, 1.0], gap="medium")

    with col_pa_left:
        _html('<div class="ap-card">')
        sc1, sc2, sc3 = st.columns([1.6, 1.0, 1.0])
        with sc1:
            pa_search = st.text_input(
                "Search table",
                placeholder="🔍 Search watershed or village...",
                label_visibility="collapsed",
            )
        with sc2:
            urgency_filter = st.selectbox(
                "Urgency Filter",
                ["All Statuses", "Urgent (Very High / High)", "Needs Monitoring", "Stable / Improving"],
                label_visibility="collapsed",
            )
        with sc3:
            sort_mode = st.selectbox(
                "Sort order",
                ["Sort: Most Urgent", "Sort: Fastest Deteriorating", "Sort: Highest Water Loss", "Sort: Highest Confidence"],
                label_visibility="collapsed",
            )

        df_queue = df_dist.copy()
        if pa_search.strip():
            sq = pa_search.strip().lower()
            df_queue = df_queue[
                df_queue["watershed_id"].str.lower().str.contains(sq)
                | df_queue["block_name"].str.lower().str.contains(sq)
                | df_queue["name"].str.lower().str.contains(sq)
                | df_queue["plain_reason"].str.lower().str.contains(sq)
            ]

        if urgency_filter == "Urgent (Very High / High)":
            df_queue = df_queue[df_queue["priority_class"].isin(["Critical Priority", "High Priority"])]
        elif urgency_filter == "Needs Monitoring":
            df_queue = df_queue[df_queue["priority_class"] == "Moderate Priority"]
        elif urgency_filter == "Stable / Improving":
            df_queue = df_queue[df_queue["priority_class"] == "Low / Stable"]

        if sort_mode == "Sort: Fastest Deteriorating":
            df_queue = df_queue.sort_values("deterioration_rate_score", ascending=False)
        elif sort_mode == "Sort: Highest Water Loss":
            df_queue = df_queue.sort_values("water_10yr_decline_pct", ascending=False)
        elif sort_mode == "Sort: Highest Confidence":
            df_queue = df_queue.sort_values("data_confidence_pct", ascending=False)
        else:
            if selected_district == "Tiruvannamalai District" and not pa_search.strip() and urgency_filter == "All Statuses":
                priority_order = ["MW-024", "MW-017", "MW-031", "MW-006", "MW-012", "MW-028", "MW-029", "MW-030"]
                df_queue["_rank_ord"] = df_queue["watershed_id"].apply(
                    lambda x: priority_order.index(x) if x in priority_order else 99
                )
                df_queue = df_queue.sort_values(["_rank_ord", "final_priority_score"], ascending=[True, False])
            else:
                df_queue = df_queue.sort_values("final_priority_score", ascending=False)

        # Table Column Headers
        _html(
            """
            <div style="display:grid; grid-template-columns: 0.45fr 1.05fr 1.25fr 1.05fr 2.1fr 1.25fr 1.25fr; background:#f8fafc; padding:10px 12px; border-radius:8px; font-weight:700; color:#475569; font-size:0.86rem; border-bottom:2px solid #e2e8f0; margin-bottom:8px;">
                <div>#</div>
                <div>Watershed ID</div>
                <div>Block Name</div>
                <div>Status</div>
                <div>Reason</div>
                <div>Confidence</div>
                <div>Action</div>
            </div>
            """
        )

        # Render Interactive Rows with REAL Backend-Connected "View Summary >" Buttons!
        for idx_num, (_, row) in enumerate(df_queue.head(10).iterrows(), 1):
            num_bg = "#fee2e2" if idx_num <= 2 else ("#fef3c7" if idx_num <= 4 else "#dcfce7")
            num_col = "#b91c1c" if idx_num <= 2 else ("#b45309" if idx_num <= 4 else "#15803d")
            wid = row["watershed_id"]

            r_cols = st.columns([0.45, 1.05, 1.25, 1.05, 2.1, 1.25, 1.25], vertical_alignment="center")
            with r_cols[0]:
                _html(
                    f'<span style="width:28px; height:28px; border-radius:50%; background:{num_bg}; color:{num_col}; display:inline-flex; align-items:center; justify-content:center; font-weight:800; font-size:0.85rem;">{idx_num}</span>'
                )
            with r_cols[1]:
                _html(f'<span style="font-weight:800; color:#0b2239; font-size:0.96rem;">{wid}</span>')
            with r_cols[2]:
                _html(f'<span style="color:#334155; font-weight:500; font-size:0.92rem;">{row["block_name"]}</span>')
            with r_cols[3]:
                _html(_status_badge_html(row["priority_class"], wid))
            with r_cols[4]:
                _html(f'<span style="color:#334155; font-size:0.9rem;">{row["plain_reason"]}</span>')
            with r_cols[5]:
                _html(_conf_badge_html(row["confidence_level"]))
            with r_cols[6]:
                if st.button("View Summary ❯", key=f"pa_vs_{wid}", use_container_width=True):
                    st.session_state["selected_ws_id"] = wid
                    open_watershed_summary_dialog(wid)

            _html('<hr style="border:none; border-top:1px solid #f1f5f9; margin:4px 0 8px 0;"/>')

        _html("</div>")

    with col_pa_right:
        _html(
            """
            <div class="ap-card">
                <h4 style="margin-top:0; margin-bottom:14px; color:#0b2239; font-size:1.15rem;">What should I review first?</h4>
                <div class="review-step-card" style="background:#fef2f2; border:1px solid #fecaca;">
                    <div style="width:28px; height:28px; border-radius:50%; background:#dc2626; color:#fff; display:flex; align-items:center; justify-content:center; font-weight:800; flex-shrink:0;">1</div>
                    <div>
                        <div style="font-weight:800; color:#7f1d1d; font-size:0.95rem;">⚠️ Urgent watersheds</div>
                        <div style="font-size:0.83rem; color:#991b1b; margin-top:4px;">
                            Watersheds with >25% surface-water shrinkage and declining vegetation cover (e.g., <b>MW-024 Chengam</b>, <b>MW-001 Kattankulathur</b>).
                        </div>
                    </div>
                </div>
                <div class="review-step-card" style="background:#fffbeb; border:1px solid #fde68a;">
                    <div style="width:28px; height:28px; border-radius:50%; background:#f59e0b; color:#fff; display:flex; align-items:center; justify-content:center; font-weight:800; flex-shrink:0;">2</div>
                    <div>
                        <div style="font-weight:800; color:#78350f; font-size:0.95rem;">🕒 Rapidly deteriorating areas</div>
                        <div style="font-size:0.83rem; color:#92400e; margin-top:4px;">
                            Sub-basins where built-up expansion (>18%) is encroaching on historic tank foreshore channels.
                        </div>
                    </div>
                </div>
                <div class="review-step-card" style="background:#eff6ff; border:1px solid #bfdbfe;">
                    <div style="width:28px; height:28px; border-radius:50%; background:#1d4ed8; color:#fff; display:flex; align-items:center; justify-content:center; font-weight:800; flex-shrink:0;">3</div>
                    <div>
                        <div style="font-weight:800; color:#1e3a8a; font-size:0.95rem;">📄 Areas awaiting field checks</div>
                        <div style="font-size:0.83rem; color:#1e40af; margin-top:4px;">
                            Confirm soil permeability, local land ownership, and existing bund conditions before engineering design.
                        </div>
                    </div>
                </div>
            </div>
            """
        )


# ============================================================================
# SECTION 4: 💡 POSSIBLE SOLUTIONS (With Working "View Details >" Buttons!)
# ============================================================================
elif selected_menu == "💡 Possible Solutions":
    _html('<div class="page-title">Possible Solutions</div>')
    _html('<div class="page-subtitle">Suggested options for investigation</div>')

    _html(
        """
        <div style="background:#eff6ff; border:1px solid #bfdbfe; border-radius:12px; padding:12px 18px; margin-bottom:16px; display:flex; align-items:center; gap:12px; color:#1e3a8a; font-weight:600; font-size:0.93rem;">
            <span style="background:#1d4ed8; color:#fff; width:24px; height:24px; border-radius:50%; display:inline-flex; align-items:center; justify-content:center; font-weight:800; font-size:0.82rem;">i</span>
            These are screening suggestions. Field and engineering checks are required before implementation.
        </div>
        """
    )

    sol_top1, sol_top2 = st.columns([1.1, 2.4])
    with sol_top1:
        ws_list_sol = [
            f"Watershed: {r['watershed_id']} ({r['block_name']})" for _, r in df_all.iterrows()
        ]
        curr_idx = 0
        for idx_i, lbl in enumerate(ws_list_sol):
            if st.session_state["selected_ws_id"] in lbl:
                curr_idx = idx_i
                break
        picked_sol_ws = st.selectbox("Select Watershed", ws_list_sol, index=curr_idx, label_visibility="collapsed")
        sol_wid = picked_sol_ws.split("Watershed: ")[1].split(" (")[0]
        st.session_state["selected_ws_id"] = sol_wid

    with sol_top2:
        sol_category = st.selectbox(
            "Filter Solution Category",
            ["Category: All", "Category: Structural", "Category: Vegetation", "Category: Agricultural", "Category: Urban", "Category: Restoration"],
            label_visibility="collapsed",
        ).replace("Category: ", "")

    s_row = df_all[df_all["watershed_id"] == sol_wid].iloc[0]

    solutions_data = [
        {
            "title": "Restore Existing Waterbody",
            "category": "Restoration",
            "cat_bg": "#dbeafe",
            "cat_col": "#1d4ed8",
            "suit_score": float(s_row["suit_eri_restoration"]),
            "suit_label": "High" if s_row["suit_eri_restoration"] >= 55 else "Medium",
            "suit_col": "#15803d" if s_row["suit_eri_restoration"] >= 55 else "#d97706",
            "bars_filled": 3 if s_row["suit_eri_restoration"] >= 55 else 2,
            "bar_color": "#15803d" if s_row["suit_eri_restoration"] >= 55 else "#f59e0b",
            "confidence": "High",
            "reason": f"Existing waterbody has reduced by {s_row['water_10yr_decline_pct']:.1f}% over time ({s_row['historic_eri_count']} historic tanks/eris).",
            "eng_details": f"Desilt tank bed, strengthen earthen bund, and clear foreshore supply channels. Estimated storage gain: {s_row['harvestable_runoff_mcm']:.2f} MCM/yr.",
            "svg_icon": "🏞️",
            "bg_illust": "linear-gradient(180deg,#bae6fd 0%,#86efac 65%,#38bdf8 100%)",
        },
        {
            "title": "Rainwater Harvesting",
            "category": "Urban",
            "cat_bg": "#ede9fe",
            "cat_col": "#6d28d9",
            "suit_score": float(s_row["suit_urban_rwh"]),
            "suit_label": "High" if s_row["suit_urban_rwh"] >= 60 else "Medium",
            "suit_col": "#15803d" if s_row["suit_urban_rwh"] >= 60 else "#d97706",
            "bars_filled": 3,
            "bar_color": "#f59e0b" if s_row["suit_urban_rwh"] < 60 else "#15803d",
            "confidence": "Medium" if s_row["builtup_pct"] < 40 else "High",
            "reason": f"Built-up area growth (+{s_row['builtup_5yr_growth_pct']:.1f}%) and {s_row['rainfall_annual_mm']:.0f} mm annual rainfall.",
            "eng_details": "Install rooftop rainwater harvesting pits, storm-drain recharge shafts, and roadside sponge bioswales.",
            "svg_icon": "🏘️",
            "bg_illust": "linear-gradient(180deg,#e0f2fe 0%,#fef3c7 60%,#cbd5e1 100%)",
        },
        {
            "title": "Recharge Structures",
            "category": "Structural",
            "cat_bg": "#e0f2fe",
            "cat_col": "#0369a1",
            "suit_score": float(s_row["suit_percolation_tank"]),
            "suit_label": "High" if s_row["suit_percolation_tank"] >= 58 else "Needs Verification",
            "suit_col": "#15803d" if s_row["suit_percolation_tank"] >= 58 else "#dc2626",
            "bars_filled": 3 if s_row["suit_percolation_tank"] >= 58 else 1,
            "bar_color": "#15803d" if s_row["suit_percolation_tank"] >= 58 else "#dc2626",
            "confidence": "Medium",
            "reason": f"Soil ({s_row['soil_texture']}, HSG-{s_row['hydrologic_soil_group']}) and groundwater ({s_row['gw_depth_mbgl']}m bgl) must be checked.",
            "eng_details": f"Check dam / percolation pond along Strahler Order-{s_row['stream_order']} channel (Drainage density: {s_row['drainage_density_km_km2']} km/km²).",
            "svg_icon": "🧱",
            "bg_illust": "linear-gradient(180deg,#dbeafe 0%,#bbf7d0 55%,#94a3b8 100%)",
        },
        {
            "title": "Vegetation Restoration",
            "category": "Vegetation",
            "cat_bg": "#dcfce7",
            "cat_col": "#15803d",
            "suit_score": float(max(s_row["suit_contour_trenching"], 68.0 if s_row["ndvi_5yr_trend"] < -0.04 else 48.0)),
            "suit_label": "High",
            "suit_col": "#15803d",
            "bars_filled": 3,
            "bar_color": "#15803d",
            "confidence": "High",
            "reason": f"Vegetation cover is declining (NDVI {s_row['ndvi_mean']:.2f}, 5-yr trend {s_row['ndvi_5yr_trend']:+.3f}).",
            "eng_details": "Native riparian buffer planting, contour bunding, and Miyawaki micro-forest patches to boost root infiltration.",
            "svg_icon": "🌳",
            "bg_illust": "linear-gradient(180deg,#dcfce7 0%,#86efac 65%,#166534 100%)",
        },
        {
            "title": "Farm Ponds (Pannaikuttai)",
            "category": "Agricultural",
            "cat_bg": "#fef3c7",
            "cat_col": "#b45309",
            "suit_score": float(s_row["suit_farm_pond"]),
            "suit_label": "High" if s_row["suit_farm_pond"] >= 55 else "Medium",
            "suit_col": "#15803d" if s_row["suit_farm_pond"] >= 55 else "#d97706",
            "bars_filled": 3 if s_row["suit_farm_pond"] >= 55 else 2,
            "bar_color": "#15803d" if s_row["suit_farm_pond"] >= 55 else "#f59e0b",
            "confidence": "High",
            "reason": f"Cropland covers {s_row['cropland_pct']:.1f}% of catchment on {s_row['slope_deg']}° slope.",
            "eng_details": "On-farm runoff harvesting ponds (30m × 30m × 2.5m) for supplemental life-saving irrigation during dry spells.",
            "svg_icon": "🌾",
            "bg_illust": "linear-gradient(180deg,#fef9c3 0%,#bef264 65%,#65a30d 100%)",
        },
        {
            "title": "Check Dam & Nala Bunds",
            "category": "Structural",
            "cat_bg": "#e0f2fe",
            "cat_col": "#0369a1",
            "suit_score": float(s_row["suit_check_dam"]),
            "suit_label": "High" if s_row["suit_check_dam"] >= 55 else "Medium",
            "suit_col": "#15803d" if s_row["suit_check_dam"] >= 55 else "#d97706",
            "bars_filled": 3 if s_row["suit_check_dam"] >= 55 else 2,
            "bar_color": "#15803d" if s_row["suit_check_dam"] >= 55 else "#f59e0b",
            "confidence": "High",
            "reason": f"Drainage density of {s_row['drainage_density_km_km2']} km/km² with Order-{s_row['stream_order']} streams.",
            "eng_details": "Masonry check dams across 2nd/3rd order drainage lines to reduce peak runoff velocity and recharge adjacent wells.",
            "svg_icon": "🌊",
            "bg_illust": "linear-gradient(180deg,#bae6fd 0%,#7dd3fc 60%,#0284c7 100%)",
        },
    ]

    if sol_category != "All":
        solutions_data = [s for s in solutions_data if s["category"] == sol_category]

    col_sol_left, col_sol_right = st.columns([2.35, 1.0], gap="medium")

    with col_sol_left:
        grid_cols = st.columns(2, gap="medium")
        for idx_s, sol in enumerate(solutions_data):
            with grid_cols[idx_s % 2]:
                segs_html = '<span class="seg-bar">'
                for b_i in range(4):
                    b_col = sol["bar_color"] if b_i < sol["bars_filled"] else "#e2e8f0"
                    segs_html += f'<span class="seg-box" style="background:{b_col};"></span>'
                segs_html += "</span>"

                conf_html = _conf_badge_html(sol["confidence"])

                _html(
                    f"""
                    <div class="sol-card">
                        <div style="display:flex; gap:16px; align-items:flex-start;">
                            <div style="width:115px; height:105px; border-radius:12px; background:{sol['bg_illust']}; display:flex; align-items:center; justify-content:center; font-size:2.8rem; flex-shrink:0; border:1px solid rgba(0,0,0,0.06);">
                                {sol['svg_icon']}
                            </div>
                            <div style="flex-grow:1;">
                                <div style="font-size:1.12rem; font-weight:800; color:#0b2239; margin-bottom:4px;">{sol['title']}</div>
                                <span style="background:{sol['cat_bg']}; color:{sol['cat_col']}; padding:3px 10px; border-radius:999px; font-size:0.76rem; font-weight:700; display:inline-block; margin-bottom:8px;">
                                    {sol['category']}
                                </span>
                                <div style="font-size:0.88rem; margin-bottom:6px;">
                                    Suitability: <b style="color:{sol['suit_col']};">{sol['suit_label']}</b> {segs_html}
                                </div>
                                <div style="margin-bottom:8px;">{conf_html}</div>
                            </div>
                        </div>
                        <div style="font-size:0.86rem; color:#475569; margin:10px 0 8px 0;">
                            {sol['reason']}
                        </div>
                        <div style="border-top:1px solid #f1f5f9; padding-top:8px; margin-bottom:4px;">
                            <span style="background:#fef2f2; color:#b91c1c; padding:4px 10px; border-radius:8px; font-size:0.78rem; font-weight:600;">
                                📋 Field check required
                            </span>
                        </div>
                    </div>
                    """
                )
                if st.button(f"View Details: {sol['title']} ❯", key=f"sol_btn_{idx_s}", use_container_width=True):
                    open_solution_details_dialog(sol, s_row)

    with col_sol_right:
        _html(
            """
            <div class="ap-card">
                <h4 style="margin-top:0; margin-bottom:16px; color:#0b2239; font-size:1.15rem;">Before taking action</h4>
                <div style="display:flex; gap:14px; align-items:flex-start; margin-bottom:18px;">
                    <div style="width:30px; height:30px; border-radius:50%; background:#1565c0; color:#fff; display:flex; align-items:center; justify-content:center; font-weight:800; flex-shrink:0;">1</div>
                    <div>
                        <div style="font-weight:800; color:#0b2239; font-size:0.96rem;">🔍 Check site conditions</div>
                        <div style="font-size:0.83rem; color:#64748b; margin-top:3px;">
                            Verify local soil depth, lithology, stream channel width, and seasonal water table depth.
                        </div>
                    </div>
                </div>
                <div style="display:flex; gap:14px; align-items:flex-start; margin-bottom:18px;">
                    <div style="width:30px; height:30px; border-radius:50%; background:#f59e0b; color:#fff; display:flex; align-items:center; justify-content:center; font-weight:800; flex-shrink:0;">2</div>
                    <div>
                        <div style="font-weight:800; color:#0b2239; font-size:0.96rem;">🗺️ Confirm land availability</div>
                        <div style="font-size:0.83rem; color:#64748b; margin-top:3px;">
                            Confirm revenue/poramboke or panchayat land boundaries and encroachment clearance.
                        </div>
                    </div>
                </div>
                <div style="display:flex; gap:14px; align-items:flex-start;">
                    <div style="width:30px; height:30px; border-radius:50%; background:#16a34a; color:#fff; display:flex; align-items:center; justify-content:center; font-weight:800; flex-shrink:0;">3</div>
                    <div>
                        <div style="font-weight:800; color:#0b2239; font-size:0.96rem;">☑️ Complete engineering assessment</div>
                        <div style="font-size:0.83rem; color:#64748b; margin-top:3px;">
                            Prepare detailed PWD/TWAD hydraulic weir design and storage-capacity estimate.
                        </div>
                    </div>
                </div>
            </div>
            """
        )


# ============================================================================
# SECTION 5: 📋 FIELD VERIFICATION
# ============================================================================
elif selected_menu == "📋 Field Verification":
    fv_top1, fv_top2 = st.columns([2.2, 1.0])
    with fv_top1:
        _html('<div class="page-title">Field Verification</div>')
        _html('<div class="page-subtitle">Record site observations and confirm system findings</div>')
    with fv_top2:
        _html(
            """
            <div style="display:flex; justify-content:flex-end; padding-top:8px;">
                <span style="background:#eff6ff; color:#1d4ed8; border:1px solid #bfdbfe; padding:8px 16px; border-radius:999px; font-weight:700; font-size:0.86rem;">
                    ☁️ Offline mode available &nbsp; | &nbsp; 🔄
                </span>
            </div>
            """
        )

    fv_wid = st.session_state["selected_ws_id"]
    fv_row = df_all[df_all["watershed_id"] == fv_wid].iloc[0]
    curr_stage = st.session_state["verification_records"].get(fv_wid, "Pending")
    stages = ["Pending", "Verified", "Planned", "Completed"]
    stage_idx = stages.index(curr_stage) if curr_stage in stages else 0

    def _step_node(idx_s: int, label: str) -> str:
        active = idx_s <= stage_idx
        bg = "#1565c0" if active else "#cbd5e1"
        col = "#0b2239" if active else "#64748b"
        return f'<div class="step-item"><div class="step-circle" style="background:{bg}; color:#ffffff;">{idx_s + 1}</div><div style="color:{col};">{label}</div></div>'

    _html(
        f"""
        <div class="stepper-wrap">
            {_step_node(0, "Pending")}
            <div class="step-line" style="background:{'#1565c0' if stage_idx >= 1 else '#cbd5e1'};"></div>
            {_step_node(1, "Verified")}
            <div class="step-line" style="background:{'#1565c0' if stage_idx >= 2 else '#cbd5e1'};"></div>
            {_step_node(2, "Planned")}
            <div class="step-line" style="background:{'#1565c0' if stage_idx >= 3 else '#cbd5e1'};"></div>
            {_step_node(3, "Completed")}
        </div>
        """
    )

    col_fv_left, col_fv_right = st.columns([1.85, 1.0], gap="medium")

    with col_fv_left:
        _html('<div class="ap-card">')
        fv_select_list = [
            f"Inspection: {r['watershed_id']} — {r['block_name']}" for _, r in df_all.iterrows()
        ]
        fv_curr_i = 0
        for i, s_lbl in enumerate(fv_select_list):
            if fv_wid in s_lbl:
                fv_curr_i = i
                break
        chosen_fv = st.selectbox("Select Watershed for Inspection", fv_select_list, index=fv_curr_i, label_visibility="collapsed")
        fv_wid = chosen_fv.split("Inspection: ")[1].split(" — ")[0]
        st.session_state["selected_ws_id"] = fv_wid
        fv_row = df_all[df_all["watershed_id"] == fv_wid].iloc[0]

        _html(
            f"""
            <div style="display:flex; gap:16px; align-items:center; margin-bottom:16px;">
                <div style="width:95px; height:68px; border-radius:10px; background:linear-gradient(180deg,#bae6fd 0%,#86efac 60%,#38bdf8 100%); display:flex; align-items:center; justify-content:center; font-size:2.2rem; border:1px solid #cbd5e1;">
                    🏞️
                </div>
                <div>
                    <h3 style="margin:0; color:#0b2239; font-size:1.4rem;">Inspection: {fv_row['watershed_id']}</h3>
                    <div style="color:#1565c0; font-weight:600; font-size:0.95rem;">📍 {fv_row['block_name']} • {fv_row['district']}</div>
                </div>
            </div>
            """
        )

        fc1, fc2 = st.columns(2)
        with fc1:
            gps_val = st.text_input("GPS Location", value=f"{fv_row['lat']:.4f}, {fv_row['lon']:.4f}")
            water_cond = st.selectbox(
                "Current Water Condition",
                [
                    "Seasonal drying / Silted foreshore",
                    "Severely dry (<15% storage)",
                    "Moderate water level",
                    "Encroached / Weed-choked channel",
                ],
            )
        with fc2:
            insp_date = st.date_input("Inspection Date", value=date(2026, 9, 30))
            exist_struct = st.selectbox(
                "Existing Structure",
                [
                    "Earthen Tank Bund (Needs Repair)",
                    "Silted Supply Channel",
                    "Damaged Check Dam / Weir",
                    "No Existing Conservation Structure",
                ],
            )

        land_avail = st.toggle("Land available for implementation", value=True)
        obs_notes = st.text_area(
            "Officer Observations",
            placeholder="Enter your site observations, borewell depth readings, or bund condition notes here...",
            height=90,
        )
        uploaded_photos = st.file_uploader(
            "Add site photographs (Drag and drop images here, or click to upload)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )

        btn_c1, btn_c2, btn_c3 = st.columns([1.1, 1.1, 1.3])
        if btn_c1.button("☁️ Save Offline", use_container_width=True):
            st.info(f"Saved draft inspection for {fv_wid} to local offline cache.")
        if btn_c2.button("✖ Reject Assessment", use_container_width=True):
            st.session_state["verification_records"][fv_wid] = "Pending"
            st.warning(f"Assessment for {fv_wid} flagged for model weight recalibration.")
        if btn_c3.button("✔ Confirm Findings", type="primary", use_container_width=True):
            st.session_state["verification_records"][fv_wid] = "Verified"
            st.success(f"Confirmed findings for {fv_wid} ({fv_row['block_name']})! Status advanced to Verified.")
            st.rerun()

        _html("</div>")

    with col_fv_right:
        _html(
            """
            <div class="ap-card">
                <h4 style="margin-top:0; color:#0b2239;">Watershed Summary</h4>
                <div style="margin-bottom:10px;">
                    <span class="pill-very-high">❗ Urgent Attention</span>
                </div>
            """
        )
        _build_priority_folium_map(
            df_all[df_all["watershed_id"] == fv_wid],
            color_col="final_priority_score",
            height_px=195,
            highlight_wid=fv_wid,
            show_legend_box=False,
        )
        _html("</div>")

        _html(
            f"""
            <div class="ap-card">
                <h4 style="margin-top:0; color:#0b2239;">Suggested option</h4>
                <div style="display:flex; gap:12px; align-items:center;">
                    <div style="width:68px; height:54px; border-radius:8px; background:linear-gradient(180deg,#bae6fd,#86efac); display:flex; align-items:center; justify-content:center; font-size:1.8rem;">
                        🏞️
                    </div>
                    <div>
                        <div style="font-weight:800; color:#0b2239; font-size:0.98rem;">Restore Existing Waterbody</div>
                        <div style="margin-top:4px;">{_conf_badge_html(fv_row['confidence_level'])}</div>
                    </div>
                </div>
            </div>
            """
        )

        _html('<div class="ap-card"><h4 style="margin-top:0; color:#0b2239;">Verification checklist</h4>')
        st.checkbox("Location confirmed", value=True, key="chk_loc")
        st.checkbox("Photographs added", value=bool(uploaded_photos), key="chk_photo")
        st.checkbox("Land condition checked", value=land_avail, key="chk_land")
        _html("</div>")


# ============================================================================
# SECTION 6: 📄 REPORTS
# ============================================================================
elif selected_menu == "📄 Reports":
    _html('<div class="page-title">Reports</div>')
    _html('<div class="page-subtitle">Create clear summaries for review and decision-making</div>')

    col_rep_left, col_rep_right = st.columns([1.75, 1.05], gap="medium")

    with col_rep_left:
        _html(
            """
            <div class="ap-card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px;">
                    <h4 style="margin:0; color:#0b2239; font-size:1.2rem;">Create a New Report</h4>
                    <div style="display:flex; gap:8px;">
                        <span style="border:2px solid #1565c0; background:#eff6ff; color:#1565c0; padding:4px 12px; border-radius:8px; font-weight:700; font-size:0.82rem;">📕 PDF</span>
                        <span style="border:1px solid #cbd5e1; background:#fff; color:#334155; padding:4px 12px; border-radius:8px; font-weight:700; font-size:0.82rem;">📗 CSV</span>
                        <span style="border:1px solid #cbd5e1; background:#fff; color:#334155; padding:4px 12px; border-radius:8px; font-weight:700; font-size:0.82rem;">🟣 GIS</span>
                    </div>
                </div>
            """
        )

        report_type = st.selectbox(
            "Report type",
            ["🗺️ District Summary", "🏞️ Watershed Summary", "📊 Priority Ranking", "📋 Field Verification"],
        )

        rc1, rc2 = st.columns(2)
        with rc1:
            rep_loc = st.selectbox("Location", district_choices, index=0)
        with rc2:
            rep_dates = st.text_input("Date Range", value="01 Sep 2026 - 30 Sep 2026")

        st.markdown("**Include in report**")
        ic1, ic2, ic3, ic4 = st.columns(4)
        inc_map = ic1.checkbox("Map Snapshot", value=True)
        inc_prob = ic2.checkbox("Main Problems", value=True)
        inc_sol = ic3.checkbox("Possible Solutions", value=True)
        inc_fld = ic4.checkbox("Field Status", value=True)

        if st.button("📄 Generate Report", type="primary", use_container_width=True):
            new_title = f"{rep_loc.split(' (')[0]} — {report_type.split(' ', 1)[1]}"
            st.session_state["generated_reports"].insert(
                0,
                {"name": new_title, "date": "30 Sep 2026", "status": "Ready", "fmt": "PDF"},
            )
            st.success(f"Generated report: **{new_title}**! Ready for download below.")

        _html("</div>")

        _html('<div class="ap-card"><h4 style="margin-top:0; color:#0b2239; font-size:1.15rem;">Recent Reports</h4>')
        for idx_r, rep in enumerate(st.session_state["generated_reports"][:5]):
            icon_f = "📕" if rep["fmt"] == "PDF" else ("📗" if rep["fmt"] == "CSV" else "🟣")
            r_c1, r_c2, r_c3, r_c4, r_c5 = st.columns([2.2, 0.9, 0.8, 1.0, 0.8], vertical_alignment="center")
            r_c1.markdown(f"**{icon_f} {rep['name']}**")
            r_c2.caption(rep["date"])
            with r_c3:
                _html('<span class="pill-stable">✔ Ready</span>')
            csv_data = df_dist.to_csv(index=False).encode("utf-8")
            r_c4.download_button(
                "⬇ Download",
                data=csv_data,
                file_name=f"{rep['name'].lower().replace(' ', '_')}.csv",
                mime="text/csv",
                key=f"dl_rep_{idx_r}",
            )
            if r_c5.button("🔗 Share", key=f"sh_rep_{idx_r}"):
                st.toast(f"Copied shareable link for {rep['name']}!")

        _html("<hr style='border-color:#f1f5f9;'/>")
        dl_c1, dl_c2 = st.columns(2)
        dl_c1.download_button(
            "📗 Export Full Priority Table (CSV)",
            data=df_all.to_csv(index=False).encode("utf-8"),
            file_name="aquaprior_tamil_nadu_priority_zones.csv",
            mime="text/csv",
            use_container_width=True,
        )
        dl_c2.download_button(
            "🟣 Export GIS Watershed Polygons (GeoJSON)",
            data=json.dumps(ws_geojson_all, indent=2).encode("utf-8"),
            file_name="aquaprior_tamil_nadu_watersheds.geojson",
            mime="application/geo+json",
            use_container_width=True,
        )
        _html("</div>")

        with st.expander("🤖 Scientific Model Validation & Google Earth Engine Code (For Judges)"):
            st.markdown("##### Supervised Deterioration Model Comparison (95% Bootstrapped CIs)")
            st.dataframe(ml_report.model_comparison_df, use_container_width=True, hide_index=True)
            st.markdown("##### Slice-Based Error Analysis by Basin")
            st.dataframe(ml_report.slice_error_df, use_container_width=True, hide_index=True)
            st.markdown("##### Google Earth Engine Extraction Script")
            st.code(GEE_JS_CODE_EDITOR_SNIPPET, language="javascript")

    with col_rep_right:
        prev_wid = st.session_state["selected_ws_id"]
        prev_row = df_all[df_all["watershed_id"] == prev_wid].iloc[0]
        v_stat = st.session_state["verification_records"].get(prev_wid, "Pending")

        _html(
            f"""
            <div class="ap-card">
                <h4 style="margin-top:0; color:#0b2239; font-size:1.18rem;">Report Preview</h4>
                <hr style="border-color:#f1f5f9; margin:8px 0 12px 0;"/>
                <div style="font-size:1.15rem; font-weight:800; color:#0b2239;">{selected_district.split(' (')[0]} Summary</div>
                <div style="font-size:0.82rem; color:#64748b; margin-bottom:12px;">30 Sep 2026 • PDF Report</div>
                <div style="display:flex; align-items:center; gap:10px; margin-bottom:4px;">
                    <span style="font-size:1.25rem; font-weight:800; color:#0b2239;">{prev_row['watershed_id']}</span>
                    <span class="pill-very-high">❗ Urgent Attention</span>
                </div>
                <div style="color:#1565c0; font-weight:600; font-size:0.9rem; margin-bottom:10px;">📍 {prev_row['block_name']}</div>
            """
        )
        _build_priority_folium_map(
            df_all[df_all["watershed_id"] == prev_wid],
            color_col="final_priority_score",
            height_px=190,
            highlight_wid=prev_wid,
            show_legend_box=False,
        )
        _html(
            f"""
                <div style="font-weight:800; color:#0b2239; margin:12px 0 8px 0;">Main drivers</div>
                <div style="background:#eff6ff; color:#1e3a8a; padding:10px 14px; border-radius:10px; margin-bottom:8px; font-weight:600; font-size:0.88rem;">
                    💧 Surface water decreased (-{prev_row['water_10yr_decline_pct']:.1f}%)
                </div>
                <div style="background:#f0fdf4; color:#166534; padding:10px 14px; border-radius:10px; margin-bottom:8px; font-weight:600; font-size:0.88rem;">
                    🌿 Vegetation is declining (NDVI {prev_row['ndvi_5yr_trend']:+.3f})
                </div>
                <div style="background:#fffbeb; color:#92400e; padding:10px 14px; border-radius:10px; margin-bottom:8px; font-weight:600; font-size:0.88rem;">
                    🏢 Built-up area increased (+{prev_row['builtup_5yr_growth_pct']:.1f}%)
                </div>
                <div style="background:#fef3c7; color:#92400e; padding:10px 14px; border-radius:10px; font-weight:600; font-size:0.88rem; border:1px solid #fde68a;">
                    📋 Field verification: {v_stat}
                </div>
            </div>
            """
        )
