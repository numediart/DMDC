import os
import sys
import torch
import pandas as pd
from omegaconf import OmegaConf
from nemo.collections.asr.models import EncDecCTCModel
import shutil
import tempfile
import torchaudio

def load_parakeet_model(model_name="nvidia/parakeet-tdt-0.6b-v2"):
    print("[Parakeet] Loading model:", model_name)
    model = EncDecCTCModel.from_pretrained(
        model_name=model_name, 
        map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    model.eval()
    return model

def transcribe_audio(model, audio_path):
    # Temp copy of the file to avoid issues with file locks or permissions
    waveform, sample_rate = torchaudio.load(audio_path)

    # Vérifier si le son est stéréo (plus d'un canal)
    if waveform.shape[0] > 1:
        # Convertir en mono en moyennant les canaux
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Créer un fichier temporaire pour la version mono
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            mono_audio_path = tmp_file.name
            torchaudio.save(mono_audio_path, waveform, sample_rate)
        
        # Transcrire le fichier mono
        result = model.transcribe([mono_audio_path])
        
        # Supprimer le fichier temporaire
        os.remove(mono_audio_path)
        
    else:
        # Si déjà en mono, transcrire directement
        result = model.transcribe([audio_path])

    return result[0].text.strip()

def main():
    if len(sys.argv) < 3:
        print("Usage: python transcribe_parakeet.py <output_dir> <audio1.wav> <audio2.wav> ...")
        sys.exit(1)

    video_idx = sys.argv[1]
    audio_files = sys.argv[2:]

    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # remonte deux niveaux
    base_dir = os.path.join(project_root, "V0DataSet")
    CUSTOM_SEGMENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test", "clean_segments")

    # Puis remplacer cette ligne :
    segments_csv_path = os.path.join(base_dir, "segments", f"{video_idx}_segments.csv")
    # segments_csv_path = os.path.join(CUSTOM_SEGMENT_DIR, f"{video_idx}_clean_segments.csv")

    if not os.path.exists(segments_csv_path):
        print(f"[ERROR] Segments file not found: {segments_csv_path}")
        sys.exit(1)

    df = pd.read_csv(segments_csv_path)
    df["transcription"] = ""

    model = load_parakeet_model()

    if len(df) != len(audio_files):
        print("[ERROR] Number of audio files doesn't match number of segments.")
        sys.exit(1)

    for i, audio_path in enumerate(audio_files):
        transcription = transcribe_audio(model, audio_path)
        df.at[i, "transcription"] = transcription

    output_csv_path = segments_csv_path
    df.to_csv(output_csv_path, index=False)
    print(f"[Parakeet] Updated segments CSV: {output_csv_path}")

    tmp_wav_path = os.path.join(base_dir, "tmp_wav")
    if os.path.exists(tmp_wav_path):
        shutil.rmtree(tmp_wav_path)
        print(f"[CLEANUP] Deleted full tmp_wav directory: {tmp_wav_path}")

if __name__ == "__main__":
    main()