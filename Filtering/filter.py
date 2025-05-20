import os
import cv2
import ffmpeg
import pandas as pd
import yt_dlp
import mediapipe as mp
from tqdm import tqdm
import platform
import csv

# --- PARAMÈTRES ---
URL = "https://www.youtube.com/watch?v=Ksi9rL2sDXo"
OUTPUT_NAME = "video/video.mp4"
FRAME_SKIP = 5
FPS = 25
MIN_DURATION_SEC = 3
SEGMENTS_CSV = "segments/"
CLIPS_DIR = "clips/"
DEBUG_MODE = False  # Activer/désactiver l'affichage du débogage

# Initialisation MediaPipe
mp_face_detection = mp.solutions.face_detection
FACE_DETECTION_THRESHOLD = 0.7  # Seuil de confiance

# --- Détection automatique de ffmpeg ---
def get_ffmpeg_path():
    base_path = os.path.join(os.path.dirname(__file__), "bin", "ffmpeg")
    if platform.system() == "Windows":
        return os.path.join(base_path, "ffmpeg.exe")
    else:
        return os.path.join(base_path, "ffmpeg")

ffmpeg_path = get_ffmpeg_path()
ffmpeg_dir = os.path.dirname(ffmpeg_path)
if os.path.exists(ffmpeg_path):
    os.environ["PATH"] = os.pathsep.join([ffmpeg_dir, os.environ["PATH"]])

# Rendre le fichier exécutable sur Linux si besoin
if platform.system() != "Windows" and os.path.exists(ffmpeg_path):
    os.chmod(ffmpeg_path, 0o755)

# --- Télécharger la vidéo si absente ---
def download_youtube_video(url, output_path):
    if os.path.exists(output_path):
        print(f"Vidéo déjà présente localement : {output_path}")
        return
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
        'outtmpl': output_path,
        'quiet': False,
        'merge_output_format': 'mp4',
        'ffmpeg_location': ffmpeg_path 
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    print(f"Vidéo téléchargée : {output_path}")

# --- Fonction d'extract audio ---
def extract_audio_to_wav(video_path, wav_output_path):
    os.makedirs(os.path.dirname(wav_output_path), exist_ok=True)

    if os.path.exists(wav_output_path):
        print(f"Fichier audio déjà présent : {wav_output_path}")
        return
    try:
        (
            ffmpeg
            .input(video_path)
            .output(wav_output_path, format='wav', acodec='pcm_s16le', ac=1, ar='16000')
            .overwrite_output()
            .run(quiet=True)
        )
        print(f"Audio extrait : {wav_output_path}")
    except ffmpeg.Error as e:
        print("Erreur lors de l'extraction audio :")
        print(e.stderr.decode())

# --- Fonction améliorée de détection des visages ---
def detect_faces_mediapipe(frame):
    # Prétraitement de l'image
    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=30)  # Amélioration contraste
    frame = cv2.GaussianBlur(frame, (3, 3), 0)  # Réduction du bruit
    
    with mp_face_detection.FaceDetection(
        model_selection=1, 
        min_detection_confidence=FACE_DETECTION_THRESHOLD
    ) as face_detector:
        results = face_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if DEBUG_MODE:
            if results.detections:
                for detection in results.detections:
                    mp.solutions.drawing_utils.draw_detection(frame, detection)
            cv2.putText(frame, f"Faces: {len(results.detections) if results.detections else 0}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Debug Detection', frame)
            cv2.waitKey(1)
        
        return len(results.detections) if results.detections else 0

# --- Détection des segments avec vérification croisée ---
def detect_faces_in_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    CONFIRMATION_FRAMES = 3

    segments = []

    current_category = None
    current_start = None
    confirmation_count = 0
    frame_count = 0

    with tqdm(total=total_frames, desc="Analyse frame par frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % FRAME_SKIP == 0:
                current_time = frame_count / frame_rate
                num_faces = detect_faces_mediapipe(frame)

                # Déterminer la catégorie
                if num_faces == 1:
                    category = "single"
                elif num_faces == 2:
                    category = "dyadic"
                else:
                    category = "other"

                # Si la catégorie reste la même, incrémenter
                if category == current_category:
                    confirmation_count = min(confirmation_count + 1, CONFIRMATION_FRAMES)
                else:
                    # Nouvelle catégorie → confirmer avec des frames stables
                    if confirmation_count >= CONFIRMATION_FRAMES and current_category is not None:
                        segment_end = current_time - (CONFIRMATION_FRAMES - 1) / frame_rate
                        if segment_end - current_start >= MIN_DURATION_SEC:
                            segments.append((round(current_start, 2), round(segment_end, 2), current_category))

                    # Redémarrer une nouvelle catégorie
                    current_category = category
                    current_start = current_time
                    confirmation_count = 1

            frame_count += 1
            pbar.update(1)

    # Fin de la dernière séquence
    end_time = frame_count / frame_rate
    if confirmation_count >= CONFIRMATION_FRAMES and current_category is not None:
        if end_time - current_start >= MIN_DURATION_SEC:
            segments.append((round(current_start, 2), round(end_time, 2), current_category))

    cap.release()
    if DEBUG_MODE:
        cv2.destroyAllWindows()

    return segments

# --- Sauvegarder les segments détectés ---
def save_segments_to_csv(segments, output_csv=SEGMENTS_CSV):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df = pd.DataFrame(segments, columns=["start_time", "end_time"])
    df["duration"] = df["end_time"] - df["start_time"]
    df.to_csv(output_csv, index=False)
    print(f"Segments sauvegardés dans {output_csv}")

def export_segments_with_speaker_to_csv(segments, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["start_time", "end_time", "category", "speaker"])
        for start, end, category, speaker in segments:
            writer.writerow([f"{start:.2f}", f"{end:.2f}", category, speaker])

# --- Recharger les segments depuis un CSV ---
def load_segments_from_csv(csv_path=SEGMENTS_CSV):
    df = pd.read_csv(csv_path)
    return list(zip(df["start_time"], df["end_time"]))

# --- Couper les segments avec ffmpeg ---
def cut_video_segments(input_path, segments, output_dir=CLIPS_DIR):
    os.makedirs(output_dir, exist_ok=True)
    for i, (start, end) in enumerate(segments):
        output_clip = os.path.join(output_dir, f"clip_{i+1:03d}.mp4")
        if os.path.exists(output_clip):
            print(f"Clip déjà existant, on saute : {output_clip}")
            continue
        duration = end - start
        (
            ffmpeg
            .input(input_path, ss=start, t=duration)
            .output(output_clip, vcodec="libx264", acodec='aac')
            .overwrite_output()
            .run(quiet=True)
        )
        print(f"Segment sauvegardé : {output_clip}")


def extract_audio_clips(clips_dir, wav_output_dir):
    os.makedirs(wav_output_dir, exist_ok=True)
    for clip_file in sorted(os.listdir(clips_dir)):
        if clip_file.endswith(".mp4"):
            clip_path = os.path.join(clips_dir, clip_file)
            wav_path = os.path.join(wav_output_dir, os.path.splitext(clip_file)[0] + ".wav")
            extract_audio_to_wav(clip_path, wav_path)
