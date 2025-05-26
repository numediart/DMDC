import os
import sys
import torch
import pandas as pd
from omegaconf import OmegaConf
from nemo.collections.asr.models import EncDecCTCModel

def load_parakeet_model(model_name="nvidia/parakeet-tdt-0.6b-v2"):
    print("[Parakeet] Loading model:", model_name)
    model = EncDecCTCModel.from_pretrained(model_name=model_name, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.eval()
    return model

def transcribe_audio(model, audio_path):
    print("[Parakeet] Transcribing:", audio_path)
    result = model.transcribe([audio_path])
    return result[0]

def main():
    if len(sys.argv) < 3:
        print("Usage: python transcribe_parakeet.py <output_dir> <audio1.wav> <audio2.wav> ...")
        sys.exit(1)

    output_dir = sys.argv[1]
    audio_files = sys.argv[2:]

    os.makedirs(output_dir, exist_ok=True)

    model = load_parakeet_model()

    for audio_path in audio_files:
        transcription = transcribe_audio(model, audio_path)
        filename = os.path.splitext(os.path.basename(audio_path))[0]
        output_csv = os.path.join(output_dir, f"{filename}.csv")

        df = pd.DataFrame([{
            "start": 0.0,
            "end": -1.0,
            "text": transcription
        }])
        df.to_csv(output_csv, index=False)
        print(f"[Parakeet] Saved: {output_csv}")

if __name__ == "__main__":
    main()