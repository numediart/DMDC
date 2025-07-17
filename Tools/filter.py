import os
import sys

import cv2
import ffmpeg
import pandas as pd
import yt_dlp
import mediapipe as mp
from tqdm import tqdm
import platform
import csv
from pydub import AudioSegment
import subprocess
import time

# --- PARAMETERS ---
URL = "https://www.youtube.com/watch?v=Ksi9rL2sDXo"
OUTPUT_NAME = "video/video.mp4"
FRAME_SKIP = 5
FPS = 25
MIN_DURATION_SEC = 3
SEGMENTS_CSV = "segments/"
CLIPS_DIR = "V0DataSet/clips_video/"
DEBUG_MODE = False  # Enable/disable debug display

DISPLAY_DISABLE_LINUX=False

# MediaPipe Initialization
mp_face_detection = mp.solutions.face_detection
FACE_DETECTION_THRESHOLD = 0.7  # Confidence threshold

# --- Automatic ffmpeg detection ---
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

# Make the file executable on Linux if needed
if platform.system() != "Windows" and os.path.exists(ffmpeg_path):
    os.chmod(ffmpeg_path, 0o755)

# --- Download the video if missing ---
def download_youtube_video(url, output_path):
    if os.path.exists(output_path):
        print(f"Video already present locally: {output_path}")
        return
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
        'outtmpl': output_path,
        'quiet': False,
        'merge_output_format': 'mp4',
        'ffmpeg_location': ffmpeg_path,
        'writesubtitles': False,
        'writeautomaticsub': False,
        'postprocessors': [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4',
        }],
        'format_selector': lambda ctx: (
            ctx.get('formats') and 
            [f for f in ctx['formats'] if f.get('acodec') != 'none' and f.get('vcodec') != 'none']
        ) or ctx.get('formats', [])
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    print(f"Video downloaded: {output_path}")


def download_youtube_video_480p_h264(url, output_path):
    if os.path.exists(output_path):
        print(f"Video already present locally: {output_path}")
        
    else:
        ydl_opts = {
            'format': 'bestvideo[height<=480][vcodec=h264][ext=mp4]+bestaudio[ext=m4a]/mp4',
            'outtmpl': output_path,
            'quiet': False,
            'merge_output_format': 'mp4',
            'ffmpeg_location': ffmpeg_path,
            'writesubtitles': False,
            'writeautomaticsub': False,
            'postprocessors': [{
                'key': 'FFmpegVideoConvertor',
                'preferedformat': 'mp4',
            }],
            'format_selector': lambda ctx: (
                ctx.get('formats') and 
                [f for f in ctx['formats'] if f.get('acodec') != 'none' and f.get('vcodec') == 'h264']
            ) or ctx.get('formats', [])
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print(f"480p H264 video downloaded: {output_path}")
        
    cap = cv2.VideoCapture(output_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps        





# --- Function to extract audio ---
def extract_audio_to_wav(video_path, wav_output_path):
    os.makedirs(os.path.dirname(wav_output_path), exist_ok=True)

    if os.path.exists(wav_output_path):
        print(f"[Filter] Audio file already present: {wav_output_path}")
        return

    # Check if the video has an audio stream
    try:
        probe = ffmpeg.probe(video_path)
        audio_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'audio']
        if not audio_streams:
            print("[ERR] No audio stream found in the video.")
            return
    except ffmpeg.Error as e:
        print("[ERR] Error probing the video file:")
        print(e.stderr.decode())
        return

    try:
        (
            ffmpeg
            .input(video_path)
            .output(wav_output_path, format='wav', acodec='pcm_s16le', ac=1, ar='16000')
            .overwrite_output()
            .run(quiet=True)
        )
        print(f"[Filter] Audio extracted: {wav_output_path}")
    except ffmpeg.Error as e:
        print("[ERR] Error during audio extraction:")
        print(e.stderr.decode())

# --- Enhanced face detection function ---
def detect_faces_mediapipe(frame):
    # Image preprocessing
    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=30)  # Contrast enhancement
    frame = cv2.GaussianBlur(frame, (3, 3), 0)  # Noise reduction
    
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

# --- Detect segments with cross-checking ---
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


    if DISPLAY_DISABLE_LINUX == False:
        with tqdm(total=total_frames, desc="[Filter/Face Detection] Analyzing frame per frame ") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % FRAME_SKIP == 0:
                    current_time = frame_count / frame_rate
                    num_faces = detect_faces_mediapipe(frame)

                    # Determine the category
                    if num_faces == 1:
                        category = "single"
                    elif num_faces == 2:
                        category = "dyadic"
                    else:
                        category = "other"

                    # If the category remains the same, increment
                    if category == current_category:
                        confirmation_count = min(confirmation_count + 1, CONFIRMATION_FRAMES)
                    else:
                        # New category → confirm with stable frames
                        if confirmation_count >= CONFIRMATION_FRAMES and current_category is not None:
                            segment_end = current_time - (CONFIRMATION_FRAMES - 1) / frame_rate
                            if segment_end - current_start >= MIN_DURATION_SEC:
                                segments.append((round(current_start, 2), round(segment_end, 2), current_category))

                        # Restart a new category
                        current_category = category
                        current_start = current_time
                        confirmation_count = 1

                frame_count += 1
                pbar.update(1)
    elif DISPLAY_DISABLE_LINUX == True:
        while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % FRAME_SKIP == 0:
                    current_time = frame_count / frame_rate
                    num_faces = detect_faces_mediapipe(frame)

                    # Determine the category
                    if num_faces == 1:
                        category = "single"
                    elif num_faces == 2:
                        category = "dyadic"
                    else:
                        category = "other"

                    # If the category remains the same, increment
                    if category == current_category:
                        confirmation_count = min(confirmation_count + 1, CONFIRMATION_FRAMES)
                    else:
                        # New category → confirm with stable frames
                        if confirmation_count >= CONFIRMATION_FRAMES and current_category is not None:
                            segment_end = current_time - (CONFIRMATION_FRAMES - 1) / frame_rate
                            if segment_end - current_start >= MIN_DURATION_SEC:
                                segments.append((round(current_start, 2), round(segment_end, 2), current_category))

                        # Restart a new category
                        current_category = category
                        current_start = current_time
                        confirmation_count = 1

                frame_count += 1

    # End of the last sequence
    end_time = frame_count / frame_rate
    if confirmation_count >= CONFIRMATION_FRAMES and current_category is not None:
        if end_time - current_start >= MIN_DURATION_SEC:
            segments.append((round(current_start, 2), round(end_time, 2), current_category))

    cap.release()
    if DEBUG_MODE:
        cv2.destroyAllWindows()

    return segments

# --- Save detected segments ---
def save_segments_to_csv(segments, output_csv=SEGMENTS_CSV):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df = pd.DataFrame(segments, columns=["start_time", "end_time"])
    df["duration"] = df["end_time"] - df["start_time"]
    df.to_csv(output_csv, index=False)
    print(f"[Filter] Segments saved to {output_csv}")

def export_segments_with_speaker_to_csv(segments, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["start_time", "end_time", "category", "speaker"])
        for start, end, category, speaker in segments:
            writer.writerow([f"{start:.2f}", f"{end:.2f}", category, speaker])

# --- Reload segments from a CSV ---
def load_segments_from_csv(csv_path=SEGMENTS_CSV):
    df = pd.read_csv(csv_path)
    return list(zip(df["start_time"], df["end_time"], df["category"]))

# --- Cut segments with ffmpeg ---
def cut_video_segments(input_path, segments, output_dir=CLIPS_DIR):
    os.makedirs(output_dir, exist_ok=True)
    for i, (start, end) in enumerate(segments):
        output_clip = os.path.join(output_dir, f"{int(start*30)}_to_{int(end*30)}_segment.mp4")

        if os.path.exists(output_clip):
            print(f"[Filter/Info] Clip already exists, skipping: {output_clip}")
            continue
        duration = end - start
        (
            ffmpeg
            .input(input_path, ss=start, t=duration)
            .output(output_clip, vcodec="libx264", acodec='aac')
            .overwrite_output()
            .run(quiet=True)
        )
        print(f"[Filter] Segment saved: {output_clip}")

# --- Extract segments from a CSV ---
def extract_dyadic_clips(video_id,dataset):
    video_path = dataset+f"/mp4/{video_id}_video.mp4"
    segment_path = dataset+f"/segments/{video_id}_segments.csv"
    output_dir = os.path.join(dataset,"clips_video", f"{video_id}_video")
    
    df = pd.read_csv(segment_path)
    dyadic_df = df[df["category"] == "dyadic"]

    dyadic_segments = [(float(row["start_time"]), float(row["end_time"])) for _, row in dyadic_df.iterrows()]
    
    if not dyadic_segments:
        print(f"[Info] No dyadic segments found for video {video_id}.")
        return

    cut_video_segments(video_path, dyadic_segments, output_dir)

def split_audio_from_csv(audio_path, csv_path, output_dir):
    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    generated_files = []

    for idx, row in df.iterrows():
        start = float(row['start_time'])
        duration = float(row['end_time']) - float(row['start_time'])

        segment_name = f"segment_{idx:04d}_{int(start*1000)}ms_{int((start+duration)*1000)}ms.wav"
        segment_path = os.path.join(output_dir, segment_name)

        subprocess.run([
            "ffmpeg", "-y", "-i", audio_path,
            "-ss", str(start),
            "-t", str(duration),
            "-acodec", "copy",
            segment_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        generated_files.append(segment_path)

    return generated_files

def wait_for_file_release(path, timeout=5):
    start_time = time.time()
    while True:
        try:
            with open(path, 'rb'):
                return True
        except PermissionError:
            if time.time() - start_time > timeout:
                return False
            time.sleep(0.5)

def extract_audio_clips(clips_dir, wav_output_dir):
    os.makedirs(wav_output_dir, exist_ok=True)
    for clip_file in sorted(os.listdir(clips_dir)):
        if clip_file.endswith(".mp4"):
            clip_path = os.path.join(clips_dir, clip_file)
            wav_path = os.path.join(wav_output_dir, os.path.splitext(clip_file)[0] + ".wav")
            extract_audio_to_wav(clip_path, wav_path)
