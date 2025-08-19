import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
from utils import detect_faces_in_image, detect_faces_in_video, detect_faces_in_webcam

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Face Detection App", layout="wide")
page_bg = """
<style>
.stApp {
    background-color: #1e1e2f;
    color: white;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ---------- TITLE ----------
st.markdown("""
    <style>
    .title {
        text-align: center;
        color: white;
        background: linear-gradient(90deg, #2c3e50, #3498db);
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown(
    '<h1 class="title">😎 Face Detection App (Image | Video | Webcam)</h1>',
    unsafe_allow_html=True
)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.sidebar.image("rishu.jpg")
    st.sidebar.header("👥 About Us")
    st.sidebar.header("💬 Contact Us")
    st.sidebar.text("📞 8809972414")
    st.sidebar.text("✉️ rishabhverma190388099@gmail.com")
    st.sidebar.text("We are a group of ML engineers working on Human face detection")

# ---------- TABS ----------
tab1, tab2, tab3 = st.tabs(["🖼 Image", "🎥 Video", "📷 Webcam"])

# ---------------------- IMAGE ----------------------
with tab1:
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = np.array(Image.open(uploaded_file).convert("RGB"))
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        with col2:
            if st.button("Start Detection", key="img_detect"):
                result_img = detect_faces_in_image(image.copy())
                result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)  # fix grey issue
                st.image(result_img, caption="Detected Faces", use_container_width=True)

# ---------------------- VIDEO ----------------------
with tab2:
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        start_btn = st.button("Start Detection", key="vid_start")
        stop_btn = st.button("Stop Detection", key="vid_stop")
        stframe = st.empty()

        if start_btn:
            for frame in detect_faces_in_video(tfile.name):
                stframe.image(frame, channels="BGR", use_container_width=False, width=400)
                if stop_btn:
                    break

# ---------------------- WEBCAM ----------------------
with tab3:
    stframe = st.empty()
    run_webcam = st.checkbox("Start Webcam")   # webcam on/off control
    detect_mode = st.checkbox("Detect Faces")  # detection on/off control

    if run_webcam:
        for frame in detect_faces_in_webcam(detect=detect_mode):
            stframe.image(frame, channels="BGR", use_container_width=False, width=500)
            if not run_webcam:  # agar user ne checkbox hata diya
                break
