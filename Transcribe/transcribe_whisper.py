import os
import sys
import shutil
import tempfile
import pandas as pd
import whisper
import torch

def load_whisper_model(model_type="base.en"):
    print(f"[Whisper] Loading model: {model_type}")
    return whisper.load_model(model_type)

def transcribe_audio(model, audio_path):
    # Crée une copie temporaire pour éviter les conflits
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        temp_audio_path = tmp_file.name
    shutil.copy(audio_path, temp_audio_path)

    print(f"[Whisper] Transcribing: {audio_path}")
    result = model.transcribe(temp_audio_path)

    try:
        os.remove(temp_audio_path)
    except Exception as e:
        print(f"[WARN] Could not delete temp audio: {e}")
    
    return result["text"].strip()

def main():
    if len(sys.argv) < 3:
        print("Usage: python transcribe_whisper.py <video_idx> <audio1.wav> <audio2.wav> ...")
        sys.exit(1)

    video_idx = sys.argv[1]
    audio_files = sys.argv[2:]

    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "V0DataSet")
    segments_csv_path = os.path.join(base_dir, "segments", f"{video_idx}_segments.csv")

    if not os.path.exists(segments_csv_path):
        print(f"[ERROR] Segments file not found: {segments_csv_path}")
        sys.exit(1)

    df = pd.read_csv(segments_csv_path)
    df["transcription"] = ""

    model = load_whisper_model("base.en")

    if len(df) != len(audio_files):
        print("[ERROR] Number of audio files doesn't match number of segments.")
        sys.exit(1)

    for i, audio_path in enumerate(audio_files):
        text = transcribe_audio(model, audio_path)
        df.at[i, "transcription"] = text

    df.to_csv(segments_csv_path, index=False)
    print(f"[Whisper] Updated segments CSV: {segments_csv_path}")

    tmp_wav_path = os.path.join(base_dir, "tmp_wav")
    if os.path.exists(tmp_wav_path):
        shutil.rmtree(tmp_wav_path)
        print(f"[CLEANUP] Deleted tmp_wav directory: {tmp_wav_path}")

if __name__ == "__main__":
    main()
