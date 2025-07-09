import sys
from pathlib import Path
import torch
from scipy.io import wavfile
from scipy.signal import resample
import numpy as np
import pandas as pd
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
model.eval().to("cpu")

def transcribe(audio_path):
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

    inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    return processor.decode(predicted_ids[0])

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python transcribe_wav2vec2.py <video_id> <seg_0001.wav> ...")
        sys.exit(1)

    video_id = sys.argv[1]
    audio_files = sys.argv[2:]

    # Chemin vers CSV original des segments
    BASE_DIR = Path(__file__).resolve().parent.parent 
    CLEAN_SEGMENTS_DIR = BASE_DIR / "test" / "clean_segments"
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

    # Écris dans le même CSV (ou change le nom si tu préfères)
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Transcriptions saved in {csv_path}")
