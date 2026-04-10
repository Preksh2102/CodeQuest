import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import shap

# -----------------------
# PAGE CONFIG
# -----------------------

st.set_page_config(
    page_title="NanoHeal AI",
    page_icon="🧬",
    layout="wide"
)

# -----------------------
# GLASS UI STYLING
# -----------------------

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(1200px 800px at 15% 10%, rgba(34, 211, 238, 0.20), transparent 55%),
        radial-gradient(900px 700px at 85% 25%, rgba(167, 139, 250, 0.16), transparent 55%),
        radial-gradient(1000px 900px at 50% 95%, rgba(34, 197, 94, 0.10), transparent 60%),
        linear-gradient(135deg, #050816 0%, #070a18 35%, #020617 100%);
    color: rgba(255,255,255,0.92);
}

/* Global typography + smoothing */
html, body, [class*="css"]  {
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

/* Animated background "medical aurora" */
.stApp::before{
    content:"";
    position: fixed;
    inset: -30%;
    background:
        radial-gradient(closest-side at 30% 35%, rgba(34, 211, 238, 0.18), transparent 55%),
        radial-gradient(closest-side at 70% 55%, rgba(167, 139, 250, 0.14), transparent 60%),
        radial-gradient(closest-side at 55% 80%, rgba(34, 197, 94, 0.10), transparent 65%);
    filter: blur(18px) saturate(1.15);
    animation: floatAurora 14s ease-in-out infinite;
    pointer-events:none;
    z-index: 0;
}

@keyframes floatAurora {
    0%   { transform: translate3d(-2%, -1%, 0) rotate(-2deg) scale(1.02); opacity: .85; }
    50%  { transform: translate3d( 2%,  1%, 0) rotate( 2deg) scale(1.05); opacity: 1; }
    100% { transform: translate3d(-2%, -1%, 0) rotate(-2deg) scale(1.02); opacity: .85; }
}

/* Ensure app content is above bg layers */
section.main, [data-testid="stSidebar"], header, footer {
    position: relative;
    z-index: 1;
}

/* Liquid-glass cards */

.glass {
    position: relative;
    background: linear-gradient(180deg, rgba(255,255,255,0.10), rgba(255,255,255,0.05));
    backdrop-filter: blur(18px) saturate(1.25);
    -webkit-backdrop-filter: blur(18px) saturate(1.25);
    border-radius: 22px;
    padding: 24px 24px;
    border: 1px solid rgba(255,255,255,0.14);
    box-shadow:
        0 18px 60px rgba(0,0,0,0.45),
        inset 0 1px 0 rgba(255,255,255,0.20);
    overflow: hidden;
    transform: translateZ(0);
    transition: transform .35s ease, box-shadow .35s ease, border-color .35s ease;
}

.glass::before{
    /* specular highlight */
    content:"";
    position:absolute;
    inset: -40% -25% auto -25%;
    height: 120%;
    background: radial-gradient(closest-side, rgba(255,255,255,0.42), transparent 65%);
    transform: rotate(-12deg);
    opacity: .28;
    pointer-events:none;
}

.glass::after{
    /* subtle neon edge (browser-safe; avoids mask artifacts) */
    content:"";
    position:absolute;
    inset: 0;
    border-radius: 22px;
    pointer-events:none;
    box-shadow:
      inset 0 0 0 1px rgba(255,255,255,0.10),
      0 0 0 1px rgba(34,211,238,0.08),
      0 0 32px rgba(34,211,238,0.08),
      0 0 36px rgba(167,139,250,0.06);
}

.glass:hover{
    transform: translateY(-2px);
    border-color: rgba(255,255,255,0.22);
    box-shadow:
        0 22px 70px rgba(0,0,0,0.52),
        0 0 0 1px rgba(34,211,238,0.12),
        inset 0 1px 0 rgba(255,255,255,0.22);
}

/* Header */

.title {
    font-size: 44px;
    font-weight: 760;
    letter-spacing: -0.02em;
    line-height: 1.08;
    background: linear-gradient(90deg, rgba(34,211,238,0.95), rgba(167,139,250,0.92), rgba(34,197,94,0.88));
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    margin-top: 6px;
}

/* Futuristic subtitle look */
.subtitle {
    color: rgba(148,163,184,0.92);
    font-size: 15px;
    line-height: 1.6;
}

/* Soft "scanner" accent under header */
.scanline {
    position: relative;
    height: 10px;
    margin: 10px 0 18px 0;
    border-radius: 999px;
    background: linear-gradient(90deg, transparent, rgba(34,211,238,0.35), rgba(167,139,250,0.22), transparent);
    overflow: hidden;
}
.scanline::after{
    content:"";
    position:absolute;
    inset: 0;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.35), transparent);
    transform: translateX(-60%);
    animation: scan 2.6s ease-in-out infinite;
    opacity: .55;
}
@keyframes scan{
    0% { transform: translateX(-60%); }
    55% { transform: translateX(60%); }
    100% { transform: translateX(60%); }
}

/* Metric cards */

.glass-metric{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    min-height: 118px;
    gap: 6px;
}

.metric {
    font-size:36px;
    font-weight:700;
    letter-spacing: -0.01em;
    line-height: 1.0;
}

.metric-label {
    font-size:18px;
    color: rgba(148,163,184,0.92);
    line-height: 1.2;
}

/* Sidebar: glass panel */
[data-testid="stSidebar"] > div:first-child{
    background: linear-gradient(180deg, rgba(255,255,255,0.07), rgba(255,255,255,0.04));
    backdrop-filter: blur(16px) saturate(1.25);
    -webkit-backdrop-filter: blur(16px) saturate(1.25);
    border-right: 1px solid rgba(255,255,255,0.10);
}

/* Buttons: futuristic */
div.stButton > button{
    width: 100%;
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.16);
    background: linear-gradient(135deg, rgba(34,211,238,0.16), rgba(167,139,250,0.14));
    color: rgba(255,255,255,0.92);
    box-shadow: 0 12px 32px rgba(0,0,0,0.35);
    transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease, background .2s ease;
}
div.stButton > button:hover{
    transform: translateY(-1px);
    border-color: rgba(34,211,238,0.35);
    box-shadow: 0 16px 44px rgba(0,0,0,0.45);
}
div.stButton > button:active{
    transform: translateY(0px) scale(0.99);
}

/* Sliders: subtle glow */
[data-testid="stSlider"] [role="slider"]{
    box-shadow: 0 0 0 6px rgba(34,211,238,0.08);
}

/* Plotly charts: rounded + glassy container */
[data-testid="stPlotlyChart"] > div{
    border-radius: 18px;
    overflow: hidden;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
}

/* Chart entrance animation */
[data-testid="stPlotlyChart"]{
    animation: chartIn .55s ease both;
    transform-origin: 50% 20%;
}
@keyframes chartIn{
    from { opacity: 0; transform: translateY(10px) scale(0.992); filter: saturate(1.1); }
    to   { opacity: 1; transform: translateY(0px) scale(1);     filter: saturate(1.0); }
}

/* Cinematic alert overlay */
.alert-overlay{
    position: fixed;
    inset: 0;
    z-index: 9999;
    pointer-events: none;
    display: grid;
    place-items: center;
    padding: 22px;
}
.alert-overlay::before{
    content:"";
    position:absolute;
    inset: 0;
    background:
      radial-gradient(900px 420px at 50% 45%, rgba(239,68,68,0.18), transparent 60%),
      linear-gradient(180deg, rgba(239,68,68,0.08), rgba(0,0,0,0.0));
    animation: redPulse 0.9s ease-in-out infinite;
}
.alert-panel{
    position: relative;
    width: min(760px, 94vw);
    border-radius: 22px;
    padding: 18px 18px;
    background: linear-gradient(180deg, rgba(255,255,255,0.10), rgba(255,255,255,0.05));
    backdrop-filter: blur(16px) saturate(1.25);
    -webkit-backdrop-filter: blur(16px) saturate(1.25);
    border: 1px solid rgba(255,255,255,0.16);
    box-shadow: 0 22px 80px rgba(0,0,0,0.60), 0 0 0 1px rgba(239,68,68,0.25);
    overflow: hidden;
}
.alert-panel::after{
    content:"";
    position:absolute;
    inset: 0;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.22), transparent);
    transform: translateX(-60%);
    animation: scan 1.9s ease-in-out infinite;
    opacity: .35;
    pointer-events:none;
}
.alert-top{
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap: 12px;
    margin-bottom: 10px;
}
.alert-title{
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.95);
}
.alert-badge{
    font-size: 12px;
    padding: 6px 10px;
    border-radius: 999px;
    background: rgba(239,68,68,0.16);
    border: 1px solid rgba(239,68,68,0.35);
    color: rgba(255,255,255,0.92);
}
.alert-body{
    color: rgba(255,255,255,0.92);
    line-height: 1.5;
}
@keyframes redPulse{
    0% { opacity: .55; }
    50% { opacity: 1; }
    100% { opacity: .55; }
}

</style>
""", unsafe_allow_html=True)

# -----------------------
# LOAD MODELS
# -----------------------

@st.cache_resource
def load_models():
    healing_model = joblib.load("models/healing_model.pkl")
    infection_model = joblib.load("models/infection_model.pkl")
    return healing_model, infection_model

healing_model, infection_model = load_models()

# -----------------------
# HEADER
# -----------------------

st.markdown('<div class="title">NanoHeal AI Clinical Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="scanline"></div>', unsafe_allow_html=True)

st.markdown("""
<div class="subtitle">
AI powered wound monitoring system using <b>XGBoost regression</b> to analyze wound environment conditions and predict:
<ul style="margin: 8px 0 0 18px;">
  <li>Healing Progress</li>
  <li>Infection Risk</li>
</ul>
</div>
""", unsafe_allow_html=True)

# -----------------------
# SIDEBAR INPUT
# -----------------------

st.sidebar.header("Clinical Input")

day = st.sidebar.slider("Post-Operative Day",1,30,7)
ph = st.sidebar.slider("Wound pH",3.0,10.0,7.0)
temperature = st.sidebar.slider("Temperature (°C)",30.0,42.0,36.5)
moisture = st.sidebar.slider("Moisture %",0,100,50)

run_ai = st.sidebar.button("Run AI Prediction")

# -----------------------
# AI PREDICTION
# -----------------------

if run_ai:

    X = pd.DataFrame([{
        "day": day,
        "pH": ph,
        "temperature": temperature,
        "moisture": moisture
    }])

    healing = float(healing_model.predict(X)[0])
    infection = float(infection_model.predict(X)[0])

    # Clamp model outputs to valid percentage range
    healing = float(np.clip(healing, 0.0, 100.0))
    infection = float(np.clip(infection, 0.0, 100.0))

    # Cinematic safety alerts (simple heuristic vs day)
    expected_healing = (day / 30.0) * 100.0
    healing_too_low = healing < max(0.0, expected_healing - 15.0)
    infection_too_high = infection >= 75.0

    alert_reasons = []
    if healing_too_low:
        alert_reasons.append(f"Healing is low for Day {day} (expected ~{expected_healing:.0f}%).")
    if infection_too_high:
        alert_reasons.append("Infection risk is critically high.")

    if alert_reasons:
        st.markdown(
            f"""
            <div class="alert-overlay">
              <div class="alert-panel">
                <div class="alert-top">
                  <div class="alert-title">Clinical Warning</div>
                  <div class="alert-badge">ALERT</div>
                </div>
                <div class="alert-body">
                  {'<br/>'.join(alert_reasons)}
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.error(" / ".join(alert_reasons))

    # -----------------------
    # METRICS
    # -----------------------

    m1, m2 = st.columns(2)

    with m1:
        st.markdown(
            f"""
            <div class="glass glass-metric">
              <div class="metric">{healing:.1f}%</div>
              <div class="metric-label">Healing Progress</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with m2:
        st.markdown(
            f"""
            <div class="glass glass-metric">
              <div class="metric">{infection:.1f}%</div>
              <div class="metric-label">Infection Risk</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # -----------------------
    # GAUGES
    # -----------------------

    g1, g2 = st.columns(2)

    with g1:

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=healing,
            title={"text":"Healing Progress"},
            gauge={
                "axis":{"range":[0,100]},
                "bar":{"color":"#22c55e"},
                "steps":[
                    {"range":[0,30],"color":"#7f1d1d"},
                    {"range":[30,70],"color":"#facc15"},
                    {"range":[70,100],"color":"#16a34a"}
                ]
            }
        ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=50, b=10),
            font=dict(color="rgba(255,255,255,0.90)")
        )

        st.plotly_chart(fig,use_container_width=True)

    with g2:

        fig2 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=infection,
            title={"text":"Infection Risk"},
            gauge={
                "axis":{"range":[0,100]},
                "bar":{"color":"#ef4444"},
                "steps":[
                    {"range":[0,30],"color":"#22c55e"},
                    {"range":[30,70],"color":"#facc15"},
                    {"range":[70,100],"color":"#7f1d1d"}
                ]
            }
        ))
        fig2.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=50, b=10),
            font=dict(color="rgba(255,255,255,0.90)")
        )

        st.plotly_chart(fig2,use_container_width=True)

    # -----------------------
    # HEALING TIMELINE
    # -----------------------

    st.subheader("Healing Progress Forecast")

    days = list(range(1,15))
    predicted=[]

    for d in days:

        tempX = pd.DataFrame([{
            "day":d,
            "pH":ph,
            "temperature":temperature,
            "moisture":moisture
        }])

        predicted.append(healing_model.predict(tempX)[0])

    df = pd.DataFrame({
        "Day":days,
        "Healing %":predicted
    })

    fig3 = px.line(
        df,
        x="Day",
        y="Healing %",
        markers=True
    )
    fig3.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=45, b=10),
        font=dict(color="rgba(255,255,255,0.90)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig3.update_traces(line=dict(color="rgba(34,211,238,0.95)", width=3))

    st.plotly_chart(fig3,use_container_width=True)

    # -----------------------
    # SHAP EXPLANATION
    # -----------------------

    st.subheader("AI Decision Explanation")

    explainer = shap.TreeExplainer(healing_model)
    shap_values = explainer.shap_values(X)

    shap_df = pd.DataFrame({
        "Feature":X.columns,
        "Impact":shap_values[0]
    })

    fig4 = px.bar(
        shap_df,
        x="Impact",
        y="Feature",
        orientation="h"
    )
    fig4.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=45, b=10),
        font=dict(color="rgba(255,255,255,0.90)")
    )
    fig4.update_traces(marker_color="rgba(167,139,250,0.90)")

    st.plotly_chart(fig4,use_container_width=True)

# -----------------------
# MODEL INFO
# -----------------------

st.markdown("---")

st.markdown("""
### AI Model

This system uses **XGBoost Regressor**, a gradient boosting algorithm that combines many decision trees to model nonlinear relationships in medical data.

Inputs used by the model:

• Post-operative day  
• Wound pH level  
• Local wound temperature  
• Moisture level  

Outputs:

• Predicted healing progress (%)  
• Infection risk probability (%)
""")