import os
import sys
import torch
import pandas as pd
import whisper
import shutil
import tempfile
import torchaudio
from pathlib import Path
from tqdm import tqdm

def load_whisper_model(model_name="large"):
    print("[Whisper] Loading model:", model_name)
    model = whisper.load_model(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
    return model

def transcribe_audio(model, audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            mono_audio_path = tmp_file.name
            torchaudio.save(mono_audio_path, waveform, sample_rate)

        result = model.transcribe(mono_audio_path, fp16=torch.cuda.is_available(), language="en")
        os.remove(mono_audio_path)
    else:
        result = model.transcribe(audio_path, fp16=torch.cuda.is_available(), language="en")

    return result["text"].strip()

def main():
    if len(sys.argv) < 3:
        print("Usage: python transcribe_whisper.py <video_idx> <audio1.wav> <audio2.wav> ...")
        sys.exit(1)

    video_idx = sys.argv[1]
    audio_files = sys.argv[2:]

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_dir = os.path.join(project_root, "V0DataSet")
    CLEAN_SEGMENTS_DIR = Path(__file__).resolve().parent.parent / "test" / "clean_segments"
    segments_csv_path = CLEAN_SEGMENTS_DIR / f"{video_idx}_clean_segments.csv"

    if not os.path.exists(segments_csv_path):
        print(f"[ERROR] Segments file not found: {segments_csv_path}")
        sys.exit(1)

    df = pd.read_csv(segments_csv_path)
    df["transcription"] = ""

    model = load_whisper_model()

    if len(df) != len(audio_files):
        print("[ERROR] Number of audio files doesn't match number of segments.")
        sys.exit(1)

    print("[Whisper] Starting transcription...")
    for i, audio_path in enumerate(tqdm(audio_files, desc="Transcribing segments")):
        transcription = transcribe_audio(model, audio_path)
        df.at[i, "transcription"] = transcription

    output_csv_path = segments_csv_path
    df.to_csv(output_csv_path, index=False)
    print(f"[Whisper] Updated segments CSV: {output_csv_path}")

    tmp_wav_path = os.path.join(base_dir, "tmp_wav")
    if os.path.exists(tmp_wav_path):
        shutil.rmtree(tmp_wav_path)
        print(f"[CLEANUP] Deleted full tmp_wav directory: {tmp_wav_path}")

if __name__ == "__main__":
    main()
