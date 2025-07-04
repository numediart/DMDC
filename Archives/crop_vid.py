import cv2
import os
from PIL import Image
from facenet_pytorch import MTCNN
import torch

# === CONFIGURATION ===
VIDEO_PATH = "V0DataSet/mp4/9_video.mp4"
OUTPUT_DIR = "V0DataSet/face_extracted_mtcnn"
RESIZE_WIDTH = 640
FRAME_SKIP = 0  # mettre à 0 pour ne rien skip
MIN_FACE_SIZE = 50

# === INITIALISATION ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=True, device=device)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# === EXTRACTION DES FRAMES ===
cap = cv2.VideoCapture(VIDEO_PATH)
frame_idx = 0
face_count = 0

print("[INFO] Starting face extraction...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if FRAME_SKIP != 0 and frame_idx % FRAME_SKIP != 0:
        frame_idx += 1
        continue

    h, w = frame.shape[:2]
    scale = RESIZE_WIDTH / w
    resized_frame = cv2.resize(frame, (RESIZE_WIDTH, int(h * scale)))

    # MTCNN attend une image PIL
    img = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
    boxes, _ = mtcnn.detect(img)

    if boxes is not None:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            face = resized_frame[y1:y2, x1:x2]

            # filtrer les petits visages ou crop corrompus
            if face.shape[0] < MIN_FACE_SIZE or face.shape[1] < MIN_FACE_SIZE:
                continue

            filename = os.path.join(OUTPUT_DIR, f"frame{frame_idx:06d}_face{i}.jpg")
            cv2.imwrite(filename, face)
            face_count += 1

    frame_idx += 1

cap.release()
print(f"[INFO] Done. Extracted {face_count} faces from {frame_idx} frames.")
