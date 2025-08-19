import cv2

# Load model
model = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# ------------------ IMAGE DETECTION ------------------
def detect_faces_in_image(image, max_width=2000):
    list_faces = model.detectMultiScale(image)
    for x, y, w, h in list_faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 4)

    # ✅ Convert BGR → RGB before returning
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


# ------------------ VIDEO DETECTION ------------------
def detect_faces_in_video(video_path, max_width=600):   # video aur chhota
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        list_faces = model.detectMultiScale(frame)
        for x, y, w, h in list_faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Resize frame
        height, width = frame.shape[:2]
        if width > max_width:
            scale = max_width / width
            frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
        yield frame
    cap.release()

