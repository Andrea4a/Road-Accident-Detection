from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from demo_backend import DemoSystem, load_config, save_temp_video

st.set_page_config(page_title="Road Accident Detection Demo", layout="centered")

APP_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = APP_DIR / "config.json"

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(40, 110, 255, 0.14), transparent 26%),
            radial-gradient(circle at top right, rgba(125, 60, 255, 0.10), transparent 24%),
            linear-gradient(180deg, #08111f 0%, #0a1425 100%);
    }

    .block-container {
        max-width: 920px;
        padding-top: 2rem;
        padding-bottom: 2.5rem;
    }

    h1, h2, h3 {
        color: #f6f8fc;
    }

    .hero-title {
        text-align: center;
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin-bottom: 0.35rem;
        color: #f6f8fc;
    }

    .hero-subtitle {
        text-align: center;
        color: #9fb0c8;
        font-size: 1.02rem;
        line-height: 1.6;
        max-width: 760px;
        margin: 0 auto 1.2rem auto;
    }

    .status-wrap {
        text-align: center;
        margin: 1rem 0 1.4rem 0;
    }

    .status-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.55rem;
        padding: 0.7rem 1.15rem;
        border-radius: 999px;
        background: rgba(35, 116, 77, 0.16);
        border: 1px solid rgba(95, 220, 152, 0.24);
        color: #aaf0c6;
        font-weight: 700;
        font-size: 1rem;
    }

    .status-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #84efb4;
        display: inline-block;
    }

    .section-title {
        text-align: center;
        font-size: 1.18rem;
        font-weight: 700;
        color: #f6f8fc;
        margin: 0.2rem 0 0.85rem 0;
    }

    .section-subtitle {
        text-align: center;
        color: #9fb0c8;
        font-size: 0.98rem;
        margin-bottom: 1rem;
    }

    div[data-testid="stFileUploader"] {
        max-width: 760px;
        margin: 0 auto;
    }

    div[data-testid="stFileUploader"] section {
        background: rgba(255,255,255,0.03);
        border: 1px dashed rgba(255,255,255,0.16);
        border-radius: 20px;
        padding: 0.9rem;
    }

    .stVideo {
        border-radius: 18px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.08);
    }

    div.stButton {
        text-align: center;
        margin-top: 0.8rem;
        margin-bottom: 0.3rem;
    }

    div.stButton > button {
        min-width: 230px;
        height: 3rem;
        border-radius: 999px;
        border: none;
        background: linear-gradient(135deg, #77a7ff 0%, #4e7dff 100%);
        color: white;
        font-weight: 700;
        box-shadow: 0 10px 28px rgba(78, 125, 255, 0.25);
    }

    .result-title {
        text-align: center;
        font-size: 1.95rem;
        font-weight: 800;
        color: #f6f8fc;
        margin-top: 0.2rem;
        margin-bottom: 0.85rem;
    }

    .pill-wrap {
        text-align: center;
        margin-bottom: 1rem;
    }

    .pill {
        display: inline-flex;
        align-items: center;
        gap: 0.55rem;
        border-radius: 999px;
        padding: 0.85rem 1.2rem;
        font-weight: 800;
        font-size: 1.05rem;
    }

    .pill-red {
        background: rgba(132, 31, 48, 0.20);
        border: 1px solid rgba(255, 121, 143, 0.28);
        color: #ffb0bc;
    }

    .pill-green {
        background: rgba(34, 115, 75, 0.18);
        border: 1px solid rgba(119, 225, 164, 0.24);
        color: #aaf2c8;
    }

    .dot-red, .dot-green {
        width: 11px;
        height: 11px;
        border-radius: 50%;
        display: inline-block;
    }

    .dot-red { background: #ff8da1; }
    .dot-green { background: #88efb5; }

    .suspicious-box {
        text-align: center;
        margin: 1rem 0 1.1rem 0;
        color: #d8e1ef;
        font-size: 1rem;
    }

    .suspicious-box strong {
        color: #ffffff;
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 0.95rem 0.45rem;
        text-align: center;
    }

    div[data-testid="stMetricLabel"] {
        justify-content: center;
    }

    .divider-space {
        height: 0.6rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_system(config_path: str):
    cfg = load_config(config_path)
    return DemoSystem(cfg)


def render_header():
    st.markdown('<div class="hero-title">Road Accident Detection Demo</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Upload a video and run the automatic analysis to detect the presence of a road accident.</div>',
        unsafe_allow_html=True,
    )


def render_status():
    st.markdown(
        """
        <div class="status-wrap">
            <span class="status-chip"><span class="status-dot"></span> System loaded</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction(binary_value: int):
    st.markdown('<div class="result-title">Final result</div>', unsafe_allow_html=True)
    if binary_value == 1:
        st.markdown(
            '<div class="pill-wrap"><span class="pill pill-red"><span class="dot-red"></span> Accident detected</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="pill-wrap"><span class="pill pill-green"><span class="dot-green"></span> No accident detected</span></div>',
            unsafe_allow_html=True,
        )


render_header()

if not DEFAULT_CONFIG_PATH.exists():
    st.error("config.json not found in tesi_demo_site/config.json.")
    st.stop()

try:
    system = get_system(str(DEFAULT_CONFIG_PATH))
except Exception as exc:
    st.error(f"System loading error: {exc}")
    st.stop()

render_status()

st.markdown('<div class="section-title">Upload a video</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Supported formats: MP4, AVI, MOV, MKV, WEBM, MPEG, MPG, M4V</div>',
    unsafe_allow_html=True,
)

uploaded = st.file_uploader(
    "Upload a video",
    type=["mp4", "avi", "mov", "mkv", "webm", "mpeg", "mpg", "m4v"],
    label_visibility="collapsed",
)

if uploaded is None:
    st.stop()

st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
st.video(uploaded)

analyze = st.button("Analyze video", type="primary")

if not analyze:
    st.stop()

with st.spinner("Analysis in progress..."):
    temp_path = save_temp_video(uploaded.getvalue(), suffix=Path(uploaded.name).suffix or ".mp4")
    result = system.predict_video(temp_path)

pred = result["prediction"]
meta = result["video_meta"]

st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
render_prediction(pred["prediction_binary"])

col1, col2 = st.columns(2)
col1.metric("Confidence", f"{pred['probability']:.4f}")
col2.metric("Duration", f"{meta['duration_sec']:.2f}s")

st.markdown(
    f'<div class="suspicious-box"><strong>Most suspicious segment:</strong> {pred["best_window"]["start_sec"]:.2f}s - {pred["best_window"]["end_sec"]:.2f}s</div>',
    unsafe_allow_html=True,
)

col3, col4 = st.columns(2)
col3.metric("Video FPS", f"{meta['fps']:.2f}")
col4.metric("File", uploaded.name)

timeline_df = pd.DataFrame(pred["timeline"])
if not timeline_df.empty:
    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Anomalous score trend</div>', unsafe_allow_html=True)
    chart_df = timeline_df[["time_sec", "score"]].dropna().set_index("time_sec")
    st.line_chart(chart_df, height=280)
