import streamlit as st
import joblib
import librosa
import numpy as np
import pandas as pd

# ---------- Load model and expected feature order ----------
model = joblib.load("voice_fraud_detector.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# ---------- Feature extraction (must match training exactly) ----------
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)

    features = {}
    features['chroma_stft'] = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
    features['rms'] = np.mean(librosa.feature.rms(y=y))
    features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    features['rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(y=y))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(20):
        features[f'mfcc{i+1}'] = np.mean(mfcc[i])

    # Build a single-row DataFrame in the EXACT column order the model expects
    row = pd.DataFrame([features])
    row = row[[col for col in feature_columns if col in row.columns]]
    return row

# ---------- Page UI ----------
st.set_page_config(page_title="AI Voice Impersonation Detector", page_icon="🎙️")

st.title("🎙️ AI Voice Impersonation Detector")
st.write("Upload a voice clip to check the likelihood it's an AI-cloned / synthetic voice, "
         "and get a risk-based recommendation — a proof-of-concept for real-time fraud prevention in voice channels.")

uploaded_file = st.file_uploader("Upload a .wav audio file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    with st.spinner("Analyzing voice..."):
        # Save temporarily so librosa can read it
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_file.read())

        features_row = extract_features("temp_audio.wav")
        prob_fake = model.predict_proba(features_row)[0][1]
        risk_score = round(prob_fake * 100, 2)

    st.subheader(f"Impersonation Risk Score: {risk_score}%")
    st.progress(int(risk_score))

    if risk_score >= 70:
        st.error("🔴 HIGH RISK — Trigger call-back verification before proceeding.")
    elif risk_score >= 40:
        st.warning("🟠 MEDIUM RISK — Recommend secondary authentication (OTP / MFA).")
    else:
        st.success("🟢 LOW RISK — Voice appears genuine. Safe to proceed.")

    st.caption("⚠️ Proof-of-concept model trained on a limited public dataset (8 speakers). "
               "Production version would add multilingual Indian-accent models, real-time call streaming, "
               "and on-device inference for privacy.")
    