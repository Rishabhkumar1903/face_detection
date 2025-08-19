import streamlit as st
import cv2
import tempfile
from PIL import Image
import numpy as np
from utils import detect_faces_in_image, detect_faces_in_video
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.set_page_config(page_title="Face Detection App", layout="wide")
page_bg = """
<style>
.stApp {
    background-color: #1e1e2f;   /* 🔵 Dark bluish background */
    color: red;                  /* 🔤 Text color */
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
            background-color: black;
        }
        </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.sidebar.image("rishu.jpg")
    st.sidebar.header("👥 About US")
    
    st.sidebar.header("💬 CONTACT US")
    st.sidebar.text("📞 8809972414")
    st.sidebar.text("✉️ rishabhverma190388099@gmail.com")

    st.sidebar.text("We are a group of ML engineers working on Human face detection")


# ----------------- TABS -----------------
tab1, tab2, tab3 = st.tabs(["📷 Image Upload", "🎞️ Video Upload", "📹 Webcam (Realtime)"])

st.markdown("""
    <style>
    button[data-baseweb="tab"] {
        color: white !important;
        border-radius: 8px 8px 0px 0px !important;
    }
    button[data-baseweb="tab"]:hover {
        background-color: #222 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #111 !important;
        color: burlywood !important;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- IMAGE SECTION ----------------
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
                # ✅ Force RGB before showing
                result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                st.image(result_img, caption="Detected Faces", use_container_width=True)
# ---------------- VIDEO SECTION ----------------
with tab2:
    st.subheader("Upload a Video for Face Detection")
    uploaded_video = st.file_uploader("Choose a video", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())
        
        stframe = st.empty()
        for frame in detect_faces_in_video(tfile.name):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame, channels="RGB")

# ---------------- WEBCAM SECTION ----------------
with tab3:
    st.subheader("Realtime Face Detection with Webcam")

    class FaceDetectionProcessor(VideoProcessorBase):
        def __init__(self):
            self.model = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            faces = self.model.detectMultiScale(img, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            return frame.from_ndarray(img, format="bgr24")

    # ✅ Columns bana ke center me webcam chhota dikha
    col1, col2, col3 = st.columns([1,2,1])  # middle column 2x size
    with col2:
        webrtc_streamer(
            key="face-detection",
            video_processor_factory=FaceDetectionProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

