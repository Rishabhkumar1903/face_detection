import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
from utils import detect_faces_in_image, detect_faces_in_video, detect_faces_in_webcam

st.set_page_config(page_title="Face Detection App", layout="wide")
page_bg = """
<style>
.stApp {
    background-color: #1e1e2f;   /* 🔵 Dark bluish background */
    color: red;                /* 🔤 Text ko white kar diya */
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)


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
    '<h1 class="title" style="color:burlywood;">😎 Face Detection App (Image | Video | Webcam)</h1>', 
    unsafe_allow_html=True
)


st.markdown("""
        <style>
        [data-testid="stSidebar"] {
            background-color: black; /* black - change to any color */
        }
        </style>
    """, unsafe_allow_html=True)

with st.sidebar:
        st.sidebar.image("rishu.jpg")
        st.sidebar.header("👥About US")
        
        st.sidebar.header("💬CONTACT US")
        st.sidebar.text("📞8809972414")
        st.sidebar.text("✉️rishabhverma190388099@gmail.com")

        st.sidebar.text("We are a group of ML engineers working on Human face detection")


tab1, tab2, tab3 = st.tabs(["🖼 Image", "🎥 Video", "📷 Webcam"])

st.markdown("""
    <style>
    button[data-baseweb="tab"] {
        color: white !important;          /* Tab text color */
        border-radius: 8px 8px 0px 0px !important;
    }
    button[data-baseweb="tab"]:hover {
        background-color: #222 !important;  /* Hover effect */
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #111 !important;  /* Selected tab */
        color: burlywood !important;        /* Active tab text */
    }
    </style>
""", unsafe_allow_html=True)


# ---------------------- IMAGE ----------------------
with tab1:
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = np.array(Image.open(uploaded_file).convert("RGB"))
        col1, col2 = st.columns(2)

        # Original image
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        # Detected image
        with col2:
            if st.button("Start Detection", key="img_detect"):
                result_img = detect_faces_in_image(image.copy())

                # ✅ Convert BGR → RGB to fix grey issue
                result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

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
                stframe.image(frame, channels="BGR", use_container_width=False, width=400)  # chhota kar diya
                if stop_btn:
                    break

# ---------------------- WEBCAM ----------------------
with tab3:
    start_btn = st.button("Start Webcam", key="web_start")
    stop_btn = st.button("Stop Webcam", key="web_stop")
    stframe = st.empty()
    
    if start_btn:
        for frame in detect_faces_in_webcam():
            stframe.image(frame, channels="BGR", use_container_width=False, width=500)
            if stop_btn:
                break
