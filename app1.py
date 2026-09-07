import os
import sys
import time
import tempfile
import threading
from collections import deque
import numpy as np
import pandas as pd
import librosa
import joblib
import streamlit as st

try:
    from streamlit_webrtc import (
        webrtc_streamer,
        WebRtcMode,
        RTCConfiguration,
        AudioProcessorBase,
    )
    import av
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

st.set_page_config(
    page_title="AI Voice Impersonation & Live Call Fraud Detector",
    page_icon="🛡️",
    layout="wide"
)

st.markdown("""
<style>
    .metric-box {
        padding: 1.2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .risk-high { background-color: rgba(239, 68, 68, 0.15); border: 1px solid #ef4444; color: #f87171; }
    .risk-med  { background-color: rgba(245, 158, 11, 0.15); border: 1px solid #f59e0b; color: #fbbf24; }
    .risk-low  { background-color: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; color: #34d399; }
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_file = os.path.join(BASE_DIR, "voice_fraud_detector.pkl")
cols_file = os.path.join(BASE_DIR, "feature_columns.pkl")

if not os.path.exists(model_file):
    model_file = "voice_fraud_detector.pkl"
if not os.path.exists(cols_file):
    cols_file = "feature_columns.pkl"

if not os.path.exists(model_file) or not os.path.exists(cols_file):
    st.error("Missing `voice_fraud_detector.pkl` or `feature_columns.pkl` in working directory.")
    st.stop()

@st.cache_resource
def load_model_pipeline():
    loaded_model = joblib.load(model_file)
    cols = joblib.load(cols_file)
    return loaded_model, cols

model, expected_features = load_model_pipeline()
SAMPLE_RATE = 16000

# Dynamically resolve which probability column represents the fake/synthetic class
FAKE_CLASS_INDEX = 0

#if hasattr(model, "classes_"):
 #classes_list = [str(c).lower().strip() for c in model.classes_]
  #  for idx, c in enumerate(classes_list):
   #     if any(keyword in c for keyword in ["fake", "spoof", "synthetic", "ai", "1"]):
    ##        FAKE_CLASS_INDEX = idx
      #      break

def load_standard_audio(file_path: str, target_sr: int = 16000) -> np.ndarray:
    y, _ = librosa.load(file_path, sr=target_sr, mono=True)
    max_amp = np.max(np.abs(y))
    if max_amp > 0:
        y = y / max_amp
    return y

def extract_26_features(y: np.ndarray, sr: int = 16000) -> pd.DataFrame:
    min_length = int(sr * 0.5)
    if len(y) < min_length:
        y = np.pad(y, (0, min_length - len(y)), mode="constant")

    y_trimmed, _ = librosa.effects.trim(y, top_db=25)
    if len(y_trimmed) >= min_length:
        y = y_trimmed

    features = {
        "chroma_stft": float(np.mean(librosa.feature.chroma_stft(y=y, sr=sr))),
        "rms": float(np.mean(librosa.feature.rms(y=y))),
        "spectral_centroid": float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
        "spectral_bandwidth": float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))),
        "rolloff": float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))),
        "zero_crossing_rate": float(np.mean(librosa.feature.zero_crossing_rate(y=y))),
    }

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(20):
        features[f"mfcc{i+1}"] = float(np.mean(mfccs[i]))

    row = pd.DataFrame([features])

    if hasattr(expected_features, "__iter__"):
        cols = list(expected_features)
        row = row.reindex(columns=cols, fill_value=0.0)

    return row

def evaluate_verdict(prob_fake: float):
    score = round(prob_fake * 100.0, 2)
    if score >= 70.0:
        return score, "🔴 HIGH RISK (AI-CLONED / SYNTHETIC VOICE)", "risk-high", "Likely voice cloning impersonation. Reject telephonic transaction and execute supervisor callback verification."
    elif score >= 40.0:
        return score, "🟠 MEDIUM RISK (SUSPICIOUS PATTERNS)", "risk-med", "Unnatural acoustic markers detected. Enforce secondary multi-factor authentication (OTP/SMS)."
    else:
        return score, "🟢 LOW RISK (GENUINE HUMAN VOICE)", "risk-low", "Acoustic envelope verified genuine. Authorized to proceed."

class LiveCallRingBuffer:
    def __init__(self, sr=16000, max_seconds=3):
        self.sr = sr
        self.max_len = sr * max_seconds
        self.buffer = np.zeros(0, dtype=np.float32)
        self.lock = threading.Lock()

    def append(self, chunk: np.ndarray):
        with self.lock:
            self.buffer = np.concatenate([self.buffer, chunk])[-self.max_len:]

    def read(self) -> np.ndarray:
        with self.lock:
            return self.buffer.copy()

    def duration(self) -> float:
        with self.lock:
            return len(self.buffer) / self.sr

if WEBRTC_AVAILABLE:
    class LiveAudioProcessor(AudioProcessorBase):
        def __init__(self):
            self.ring_buffer = LiveCallRingBuffer(SAMPLE_RATE, 3)

        def recv(self, frame: "av.AudioFrame") -> "av.AudioFrame":
            try:
                sound = frame.to_ndarray()
                if sound.ndim > 1:
                    sound = np.mean(sound, axis=0)
                sound = sound.astype(np.float32)

                if np.abs(sound).max() > 1.0:
                    sound = sound / 32768.0

                if frame.sample_rate != SAMPLE_RATE:
                    sound = librosa.resample(sound, orig_sr=frame.sample_rate, target_sr=SAMPLE_RATE)

                self.ring_buffer.append(sound)
            except Exception:
                pass
            return frame

if "live_history" not in st.session_state:
    st.session_state.live_history = deque(maxlen=20)

st.title("🛡️ AI Voice Integrity Verification Engine")
st.markdown("Real-time deepfake & voice-cloning detection designed for live calls and audio auditing.")

tab_live, tab_file = st.tabs(["📞 Live On-Call Detection", "📁 Audio File Inspection"])

with tab_live:
    st.write("")
    st.markdown("### 🔴 Real-Time Live Call Stream Verifier")
    st.caption("Captures live speech directly from the microphone or connected virtual telephony loopback to score voice fraud risk continuously.")

    if not WEBRTC_AVAILABLE:
        st.error("`streamlit-webrtc` or `av` is not installed. Run `pip install streamlit-webrtc av` in terminal to use live call detection.")
    else:
        col_stream, col_monitor = st.columns([1, 1], gap="large")

        with col_stream:
            st.markdown("##### 🎙️ Audio Ingestion")
            rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
            webrtc_ctx = webrtc_streamer(
                key="live-telephony-call",
                mode=WebRtcMode.SENDONLY,
                rtc_configuration=rtc_config,
                audio_processor_factory=LiveAudioProcessor,
                media_stream_constraints={"audio": True, "video": False},
            )

        with col_monitor:
            st.markdown("##### ⚡ Live Risk Scorecard")
            metrics_display = st.empty()

            if webrtc_ctx.state.playing and webrtc_ctx.audio_processor:
                ring_buffer = webrtc_ctx.audio_processor.ring_buffer
                
                while webrtc_ctx.state.playing:
                    if ring_buffer.duration() >= 1.0:
                        try:
                            audio_window = ring_buffer.read()
                            rms_level = float(np.sqrt(np.mean(audio_window**2)))
                            
                            if rms_level > 0.005:
                                features = extract_26_features(audio_window, SAMPLE_RATE)
                                probabilities = model.predict_proba(features)[0]
                                prob_fake = float(probabilities[FAKE_CLASS_INDEX])
                                score, label, css_class, advisory = evaluate_verdict(prob_fake)
                                st.session_state.live_history.append(score)

                                with metrics_display.container():
                                    st.markdown(f'<div class="metric-box {css_class}"><h4>{label}</h4><p>{advisory}</p></div>', unsafe_allow_html=True)
                                    st.metric("Live Fraud Probability", f"{score}%")
                                    st.progress(int(score))
                                    if len(st.session_state.live_history) > 1:
                                        st.caption("Risk Trajectory over Time")
                                        st.line_chart(list(st.session_state.live_history), height=140)
                            else:
                                with metrics_display.container():
                                    st.info("🎙️ Listening... Please speak into the microphone.")
                        except Exception as err:
                            metrics_display.error(f"Inference error: {err}")
                    else:
                        metrics_display.info("⏳ Buffering incoming audio stream...")
                    
                    time.sleep(0.5)
            else:
                metrics_display.info("👉 Click **START** above and allow browser microphone permissions.")

with tab_file:
    st.write("")
    st.markdown("### 📁 File Analysis")
    uploaded = st.file_uploader("Upload recorded audio file", type=["wav", "mp3", "ogg", "flac", "m4a"])

    if uploaded:
        st.audio(uploaded)
        if st.button("Analyze Audio File", type="primary", use_container_width=True):
            with st.spinner("Decoding audio signal and running feature classifier..."):
                temp_path = None
                try:
                    ext = os.path.splitext(uploaded.name)[1].lower() or ".wav"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tf:
                        tf.write(uploaded.getvalue())
                        temp_path = tf.name

                    y = load_standard_audio(temp_path, target_sr=SAMPLE_RATE)
                    feat_row = extract_26_features(y, SAMPLE_RATE)

                    probabilities = model.predict_proba(feat_row)[0]
                    prob_fake = float(probabilities[FAKE_CLASS_INDEX])
                    score, label, css_class, advisory = evaluate_verdict(prob_fake)

                    st.markdown(f'<div class="metric-box {css_class}"><h4>{label}</h4><p>{advisory}</p></div>', unsafe_allow_html=True)
                    st.metric("Impersonation Probability Score", f"{score}%")
                    st.progress(int(score))

                    with st.expander("🛠️ Diagnostics & Extracted Features"):
                        st.write(f"**Model Classes:** `{getattr(model, 'classes_', 'N/A')}`")
                        st.write(f"**Calculated Class Probabilities:** `{probabilities}`")
                        st.write(f"**Fake Class Index Used:** `{FAKE_CLASS_INDEX}`")
                        st.dataframe(feat_row.T.rename(columns={0: "Extracted Value"}))

                except Exception as ex:
                    st.error(f"Error processing audio file: {ex}")
                finally:
                    if temp_path and os.path.exists(temp_path):
                        os.remove(temp_path)