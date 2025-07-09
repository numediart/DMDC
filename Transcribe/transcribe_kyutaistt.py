import sys
from pathlib import Path
import pandas as pd
import torch
import torchaudio
import ctranslate2
from huggingface_hub import snapshot_download

# Chemins de base
BASE_DIR = Path(__file__).resolve().parent.parent
CLEAN_SEGMENTS_DIR = BASE_DIR / "test" / "clean_segments"

# Charger le modèle
model_dir = snapshot_download("kyutai/stt-2.6b-en", local_files_only=False)
translator = ctranslate2.Translator(
    model_dir,
    compute_type="int8_float16" if torch.cuda.is_available() else "int8"
)

# Chargement audio
def load_audio(path, target_sr=16000):
    waveform, sr = torchaudio.load(path)
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    return waveform.squeeze(0).numpy().tolist()

# Transcription d'un fichier
def transcribe(audio_path):
    audio = load_audio(audio_path)
    result = translator.translate_batch([audio], task="transcribe", max_batch_size=1, beam_size=1)
    return result[0].hypotheses[0]

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python transcribe_kyutai.py <video_id> <seg_0001.wav> ...")
        sys.exit(1)

    video_id = sys.argv[1]
    audio_files = sys.argv[2:]

    csv_path = CLEAN_SEGMENTS_DIR / f"{video_id}_clean_segments.csv"
    if not csv_path.exists():
        print(f"[ERROR] CSV file not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    df["transcription"] = ""

    for idx, audio_path in enumerate(audio_files):
        print(f"[INFO] Transcribing segment {idx:04d}...")
        try:
            text = transcribe(audio_path)
            df.at[idx, "transcription"] = text
        except Exception as e:
            print(f"[ERROR] Could not transcribe {audio_path}: {e}")
            df.at[idx, "transcription"] = "ERROR"

    df.to_csv(csv_path, index=False)
    print(f"[INFO] Transcriptions saved in {csv_path}")
