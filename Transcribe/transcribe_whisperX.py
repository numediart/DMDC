import sys
from pathlib import Path
import torch
import pandas as pd
import whisperx
from whisperx.diarize import DiarizationPipeline
import gc # Garbage Collector

# 1. Configuration de WhisperX
# Détermine le périphérique (GPU si disponible, sinon CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
# Modèle de transcription ASR (ex: "large-v2", "medium", "base")
asr_model_name = "large-v2" 
# Taille du lot pour la transcription (à ajuster selon la VRAM)
batch_size = 16 
# Type de calcul (float16 est plus rapide sur GPU)
compute_type = "float16" if device == "cuda" else "int8"

# Chargement du modèle ASR Whisper
print(f"[INFO] Chargement du modèle ASR Whisper '{asr_model_name}' sur {device}...")
model = whisperx.load_model(asr_model_name, device, compute_type=compute_type, language="en")
print("[INFO] Modèle ASR chargé.")

def transcribe_whisperx(audio_path):
    """
    Transcrit un fichier audio en utilisant WhisperX avec alignement et diarisation.
    """
    # 2. Chargement de l'audio
    # WhisperX gère le ré-échantillonnage en interne.
    audio = whisperx.load_audio(audio_path)

    # 3. Transcription avec Whisper
    # La méthode 'transcribe' de WhisperX retourne des segments avec timestamps.
    result = model.transcribe(audio, batch_size=batch_size)
    
    # Si la transcription est vide, on nettoie et on retourne une chaîne vide.
    if not result["segments"]:
        gc.collect()
        torch.cuda.empty_cache()
        return ""

    # 4. Alignement du modèle (pour des timestamps plus précis au niveau des mots)
    # On charge le modèle d'alignement. 'result' contient la langue détectée.
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    # On aligne la transcription.
    aligned_result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    
    # 5. Diarisation (identification des locuteurs)
    # On charge le modèle de diarisation.
    diarize_model = DiarizationPipeline(use_auth_token="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", device=device)
    # On assigne les locuteurs aux segments alignés.
    # Le nombre de locuteurs peut être spécifié avec min_speakers et max_speakers.
    diarized_result = diarize_model(audio, min_speakers=None, max_speakers=None)
    final_result = whisperx.assign_word_speakers(diarized_result, aligned_result)

    # 6. Formatage de la sortie
    # On reconstruit la transcription complète à partir des segments, sans ajouter le nom du locuteur
    full_text = []
    for segment in final_result["segments"]:
        text = segment["text"]
        full_text.append(text)

    # Nettoyage de la mémoire GPU
    gc.collect()
    torch.cuda.empty_cache()

    return " ".join(full_text).strip()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python transcribe_whisperx.py <video_id> <seg_0001.wav> ...")
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
            text = transcribe_whisperx(audio_path)
            df.at[idx, "transcription"] = text
        except Exception as e:
            print(f"[ERROR] Could not transcribe {audio_path}: {e}")
            df.at[idx, "transcription"] = "ERROR"
            # Nettoyage en cas d'erreur pour libérer la VRAM
            gc.collect()
            torch.cuda.empty_cache()

    output_csv_path = CLEAN_SEGMENTS_DIR / f"{video_id}_clean_segments.csv"
    df.to_csv(output_csv_path, index=False)
    print(f"[INFO] Transcriptions saved in {output_csv_path}")