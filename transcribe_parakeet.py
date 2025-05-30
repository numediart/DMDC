import os
import sys
import torch
import pandas as pd
from omegaconf import OmegaConf
from nemo.collections.asr.models import EncDecCTCModel
import shutil
import tempfile

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
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        temp_audio_path = tmp_file.name
    shutil.copy(audio_path, temp_audio_path)

    print(f"[Parakeet] Transcribing copy of: {audio_path}")
    result = model.transcribe([temp_audio_path])

    # Clean up the temporary file
    try:
        os.remove(temp_audio_path)
    except Exception as e:
        print(f"[WARN] Could not delete temporary audio copy: {e}")
    return result[0].text.strip()

def main():
    if len(sys.argv) < 3:
        print("Usage: python transcribe_parakeet.py <output_dir> <audio1.wav> <audio2.wav> ...")
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