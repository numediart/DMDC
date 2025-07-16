import cv2
import os
# pip install git+https://github.com/deepinsight/insightface.git
from insightface.app import FaceAnalysis
from insightface.utils import face_align

def extract_and_align_faces(video_path, video_number,dataset):
    output_dir = f"{dataset}/face_aligned/{video_number}_video"
    os.makedirs(output_dir, exist_ok=True)

    app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0)

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    face_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = app.get(frame)

        for i, face in enumerate(faces):
            aligned = face_align.norm_crop(frame, landmark=face.kps)
            filename = os.path.join(output_dir, f"frame{frame_idx:06d}_face{i}.jpg")
            cv2.imwrite(filename, aligned)
            face_count += 1

        frame_idx += 1

    cap.release()
    print(f"[INFO] Extracted and aligned {face_count} faces from {frame_idx} frames in video {video_number}.")

# Example usage:

if __name__ == "__main__":
    extract_and_align_faces("V0DataSet/mp4/9_video.mp4", 9, "V0DataSet")