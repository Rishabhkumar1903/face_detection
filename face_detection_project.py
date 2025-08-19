import streamlit as st
import cv2
import tempfile
from PIL import Image
import numpy as np
from utils import detect_faces_in_image, detect_faces_in_video
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# -------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Face Detection App", layout="wide")
st.markdown(
    """
    <h1 style="text-align:center; color:cyan;">🚀 Face Detection App</h1>
    """,
    unsafe_allow_html=True
)

# ----------------- SIDEBAR -----------------
st.sidebar.title("Navigation")
choice = st.sidebar.radio("Choose an option", ["Image Upload", "Video Upload", "Webcam (Realtime)"])

# ---------------- IMAGE SECTION ----------------
if choice == "Image Upload":
    st.subheader("Upload an Image for Face Detection")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = np.array(Image.open(uploaded_file))
        detected = detect_faces_in_image(image)
        st.image(detected, caption="Detected Faces", use_column_width=True)

# ---------------- VIDEO SECTION ----------------
elif choice == "Video Upload":
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
elif choice == "Webcam (Realtime)":
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

    webrtc_streamer(
        key="face-detection",
        video_processor_factory=FaceDetectionProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
