import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Spotify Song Hit Prediction",
    page_icon="🎧",
    layout="wide"
)

# ---------------- SPOTIFY STYLE CSS ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #121212 0%, #000000 100%);
    color: white;
}
h1, h2, h3, h4 { color: white; }
div[data-baseweb="slider"] span { background-color: #1DB954 !important; }
.stButton>button {
    background-color: #1DB954;
    color: black;
    font-weight: bold;
    border-radius: 30px;
    height: 50px;
    font-size: 18px;
}
.stButton>button:hover { background-color: #1ed760; }
.stAlert { border-radius: 15px; }
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD FILES ----------------
df = pd.read_csv("data/songs.csv")
model = joblib.load("model/best_model.pkl")
scaler = joblib.load("model/scaler.pkl")

with open("model/metrics.json") as f:
    metrics = json.load(f)

# ---------------- TITLE ----------------
st.title("🎧 Spotify Song Hit Prediction")
st.caption("Machine Learning Project • Algorithm Comparison")

# ---------------- DATASET OVERVIEW ----------------
st.header("📊 Dataset Overview")

c1, c2, c3 = st.columns(3)
c1.metric("Total Samples", metrics["total_samples"])
c2.metric("Training Samples", metrics["train_size"])
c3.metric("Test Samples", metrics["test_size"])

train_pct = round(metrics["train_size"] / metrics["total_samples"] * 100, 2)
test_pct = round(metrics["test_size"] / metrics["total_samples"] * 100, 2)

st.caption(f"🧠 Training Data: **{train_pct}%** | 🧪 Test Data: **{test_pct}%**")

st.divider()

# ---------------- HELPER FUNCTIONS ----------------
def danceability_text(v):
    return "🕺 Hard to dance" if v < 0.3 else "🙂 Groovy" if v < 0.6 else "🔥 Club-ready"

def energy_text(v):
    return "😴 Low energy" if v < 0.3 else "⚡ Balanced" if v < 0.6 else "🚀 High energy"

def loudness_text(v):
    return "🔈 Very quiet" if v < -20 else "🔊 Studio loud" if v < -10 else "📢 Very loud"

def tempo_text(v):
    return "🐢 Slow" if v < 90 else "🚶 Medium" if v < 120 else "🏃 Fast"

def valence_text(v):
    return "😢 Sad mood" if v < 0.3 else "😐 Neutral" if v < 0.6 else "😁 Happy"

# ---------------- PREDICT NEW SONG ----------------
st.header("🎶 Predict New Song")

col1, col2 = st.columns(2)

with col1:
    danceability = st.slider("Danceability", 0.0, 1.0, 0.54)
    st.caption(danceability_text(danceability))

    energy = st.slider("Energy", 0.0, 1.0, 0.50)
    st.caption(energy_text(energy))

    loudness = st.slider("Loudness (dB)", -60.0, 0.0, -10.0)
    st.caption(loudness_text(loudness))

with col2:
    tempo = st.slider("Tempo (BPM)", 60, 200, 120)
    st.caption(tempo_text(tempo))

    valence = st.slider("Valence", 0.0, 1.0, 0.50)
    st.caption(valence_text(valence))

# ---------------- PREDICTION ----------------
st.divider()

if st.button("🎯 Predict Hit", use_container_width=True):

    input_data = np.array([[danceability, energy, loudness, tempo, valence]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.success(f"🔥 HIT SONG\n\nConfidence: **{probability*100:.2f}%**")
    else:
        st.warning(f"📉 NOT A HIT\n\nConfidence: **{(1-probability)*100:.2f}%**")

# ---------------- FOOTER ----------------
st.markdown(
    "<center style='color:#B3B3B3;'>Spotify-Inspired Song Hit Prediction System</center>",
    unsafe_allow_html=True
)
