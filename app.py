import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="AI Wound Monitoring System",
    page_icon="🧠",
    layout="wide"
)

# Load models (will work once you provide PKL files)
@st.cache_resource
def load_models():
    healing_model = joblib.load("models/healing_model.pkl")
    infection_model = joblib.load("models/infection_model.pkl")
    return healing_model, infection_model


# Header
st.title("AI-Powered Post-Operative Wound Monitoring")
st.markdown(
"""
This AI system analyzes **pH, temperature, and moisture** levels from wound environments  
to estimate **healing progress** and **infection risk** using an **XGBoost regression model**.
"""
)

# Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Sensor Values")

    day = st.slider("Post-Op Day", 1, 14, 7)
    ph = st.slider("Wound pH", 3.0, 10.0, 7.0)
    temperature = st.slider("Temperature (°C)", 30.0, 42.0, 36.5)
    moisture = st.slider("Moisture Level (%)", 0, 100, 50)

    predict = st.button("Run AI Analysis")

with col2:
    st.subheader("Predicted Results")

    if predict:
        healing_model, infection_model = load_models()

        X = np.array([[day,ph, temperature, moisture]])

        healing = healing_model.predict(X)[0]
        infection = infection_model.predict(X)[0]

        st.metric("Healing Progress", f"{healing:.1f}%")
        st.metric("Infection Risk", f"{infection:.1f}%")

        # Gauge Charts
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=healing,
            title={'text': "Healing %"},
            gauge={'axis': {'range': [0,100]}}
        ))

        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=infection,
            title={'text': "Infection Risk %"},
            gauge={'axis': {'range': [0,100]}}
        ))

        st.plotly_chart(fig2, use_container_width=True)