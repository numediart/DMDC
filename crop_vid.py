import cv2
import os
import numpy as np
from insightface.app import FaceAnalysis
from insightface.utils import face_align

# Config
VIDEO_PATH = "V0DataSet/mp4/9_video.mp4"
OUTPUT_DIR = "V0DataSet/face_aligned"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load RetinaFace
app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0)

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)
frame_idx = 0
face_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)

    for i, face in enumerate(faces):
        # Align the face to 112x112 using landmarks (RetinaFace provides them)
        aligned = face_align.norm_crop(frame, landmark=face.kps)  # kps = 5 keypoints

        # Save the aligned face
        filename = os.path.join(OUTPUT_DIR, f"frame{frame_idx:06d}_face{i}.jpg")
        cv2.imwrite(filename, aligned)
        face_count += 1

    frame_idx += 1

cap.release()
print(f"[INFO] Extracted and aligned {face_count} faces from {frame_idx} frames.")
