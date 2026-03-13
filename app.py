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
    background: linear-gradient(135deg,#0f172a,#020617);
    color:white;
}

/* Glass cards */

.glass {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(20px);
    border-radius: 20px;
    padding: 25px;
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
}

/* Header */

.title {
    font-size:42px;
    font-weight:700;
}

/* Metric cards */

.metric {
    font-size:36px;
    font-weight:700;
}

.metric-label {
    font-size:18px;
    color:#94a3b8;
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

st.markdown("""
AI powered wound monitoring system using **XGBoost regression** to analyze wound environment conditions and predict:

• Healing Progress  
• Infection Risk  
""")

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

    healing = healing_model.predict(X)[0]
    infection = infection_model.predict(X)[0]

    # -----------------------
    # METRICS
    # -----------------------

    m1, m2 = st.columns(2)

    with m1:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric">{healing:.1f}%</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Healing Progress</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with m2:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric">{infection:.1f}%</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Infection Risk</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

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