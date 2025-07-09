import sys
from pathlib import Path
import torch
from scipy.io import wavfile
from scipy.signal import resample
import numpy as np
import pandas as pd
# On peut utiliser les classes "Auto" pour plus de flexibilité
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

# --- MODIFICATIONS PRINCIPALES ICI ---
# 1. Chargement du modèle et du processeur Distil-Whisper
MODEL_ID = "distil-whisper/distil-large-v3"
torch_dtype = torch.float32 # Sur CPU, on utilise float32

print(f"[INFO] Chargement du modèle {MODEL_ID}...")
# On utilise AutoModelForSpeechSeq2Seq qui est la classe adaptée pour les modèles Whisper
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_ID, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
processor = AutoProcessor.from_pretrained(MODEL_ID)

# Le modèle est déjà optimisé pour une inférence rapide sur CPU
model.eval().to("cpu")
print("[INFO] Modèle chargé avec succès.")


def transcribe(audio_path):
    """
    Transcrit un fichier audio en utilisant le modèle Distil-Whisper.
    Le code est identique à celui pour CrisperWhisper.
    """
    target_sample_rate = 16000
    sample_rate, waveform = wavfile.read(audio_path)

    if len(waveform.shape) == 2:
        waveform = np.mean(waveform, axis=1)

    if sample_rate != target_sample_rate:
        duration = waveform.shape[0] / sample_rate
        num_samples = int(duration * target_sample_rate)
        waveform = resample(waveform, num_samples)
        sample_rate = target_sample_rate

    if waveform.dtype == np.int16:
        waveform = waveform.astype(np.float32) / 32768.0
    elif waveform.dtype == np.int32:
        waveform = waveform.astype(np.float32) / 2147483648.0
    elif waveform.dtype == np.uint8:
        waveform = (waveform.astype(np.float32) - 128) / 128.0
    else:
        waveform = waveform.astype(np.float32)
    
    # Le reste de la logique est inchangé
    inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt")
    input_features = inputs.input_features.to("cpu")

    with torch.no_grad():
        # Pour Distil-Whisper, on peut guider la génération pour de meilleurs résultats
        predicted_ids = model.generate(input_features, language="french", task="transcribe")

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python transcribe_distil_whisper.py <video_id> <seg_0001.wav> ...")
        sys.exit(1)

    video_id = sys.argv[1]
    audio_files = sys.argv[2:]

    BASE_DIR = Path(__file__).resolve().parent.parent 
    CLEAN_SEGMENTS_DIR = BASE_DIR / "test" / "clean_segments"
    csv_path = CLEAN_SEGMENTS_DIR / f"{video_id}_clean_segments.csv"
    
    if not csv_path.exists():
        print(f"[ERROR] CSV file not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    df["transcription"] = ""

    for idx, audio_path in enumerate(audio_files):
        print(f"[INFO] Transcribing segment {idx + 1}/{len(audio_files)} ({Path(audio_path).name})...")
        try:
            text = transcribe(audio_path)
            df.at[idx, "transcription"] = text.strip()
        except Exception as e:
            print(f"[ERROR] Could not transcribe {audio_path}: {e}")
            df.at[idx, "transcription"] = "ERROR"

    df.to_csv(csv_path, index=False)
    print(f"[INFO] Transcriptions saved in {csv_path}")