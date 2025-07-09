import sys
from pathlib import Path
import torch
from scipy.io import wavfile
from scipy.signal import resample
import numpy as np
import pandas as pd
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# 1. Chargement du modèle et du processeur CrisperWhisper
# Le "processor" pour Whisper combine le "feature_extractor" (pour l'audio) 
# et le "tokenizer" (pour le texte).
processor = WhisperProcessor.from_pretrained("nyrahealth/CrisperWhisper")
model = WhisperForConditionalGeneration.from_pretrained("nyrahealth/CrisperWhisper")
model.eval().to("cpu") # Vous pouvez passer sur "cuda" si vous avez un GPU

def transcribe(audio_path):
    """
    Transcrit un fichier audio en utilisant le modèle CrisperWhisper.
    """
    # La fréquence d'échantillonnage cible pour Whisper est toujours 16000 Hz.
    target_sample_rate = 16000
    
    # Votre code de lecture et de ré-échantillonnage est déjà parfait pour Whisper.
    sample_rate, waveform = wavfile.read(audio_path)

    if len(waveform.shape) == 2:
        waveform = np.mean(waveform, axis=1)

    if sample_rate != target_sample_rate:
        duration = waveform.shape[0] / sample_rate
        num_samples = int(duration * target_sample_rate)
        waveform = resample(waveform, num_samples)
        sample_rate = target_sample_rate

    # La normalisation est également correcte.
    if waveform.dtype == np.int16:
        waveform = waveform.astype(np.float32) / 32768.0
    elif waveform.dtype == np.int32:
        waveform = waveform.astype(np.float32) / 2147483648.0
    elif waveform.dtype == np.uint8:
        waveform = (waveform.astype(np.float32) - 128) / 128.0
    else:
        waveform = waveform.astype(np.float32)

    # 2. Préparation des données pour Whisper
    # Le processor s'occupe de transformer le waveform brut en "input_features".
    inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt")
    input_features = inputs.input_features.to(model.device)

    # 3. Génération de la transcription
    # Whisper utilise une méthode .generate() plutôt qu'une simple passe "forward".
    with torch.no_grad():
        predicted_ids = model.generate(input_features)

    # 4. Décodage du texte
    # On utilise le processeur pour décoder les IDs en texte.
    # skip_special_tokens=True enlève les tokens techniques (ex: <|startoftranscript|>)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    
    return transcription

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python transcribe_crisperwhisper.py <video_id> <seg_0001.wav> ...")
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
            # Whisper peut parfois retourner des espaces inutiles au début/fin.
            df.at[idx, "transcription"] = text.strip()
        except Exception as e:
            print(f"[ERROR] Could not transcribe {audio_path}: {e}")
            df.at[idx, "transcription"] = "ERROR"

    # Le nom du fichier de sortie est modifié pour ne pas écraser l'original.
    output_csv_path = CLEAN_SEGMENTS_DIR / f"{video_id}_transcriptions_crisperwhisper.csv"
    df.to_csv(output_csv_path, index=False)
    print(f"[INFO] Transcriptions saved in {output_csv_path}")