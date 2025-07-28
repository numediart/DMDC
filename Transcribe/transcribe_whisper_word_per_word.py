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

    # Extract full transcription
    full_text = result["text"].strip()

    # Extract word-by-word transcription
    word_segments = []
    for segment in result.get("segments", []):
        for word in segment.get("words", []):
            word_segments.append({
                "word": word["text"],
                "start": word["start"],
                "end": word["end"]
            })

    return full_text, word_segments


def main():

    if len(sys.argv) < 3:
        print("Usage: python transcribe_whisper_word_per_word.py <audio_files_dir> <output_dir>")
        sys.exit(1)

    audio_files_dir = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.exists(audio_files_dir):
        print(f"[ERROR] Audio files directory not found: {audio_files_dir}")
        sys.exit(1)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    audio_files = [os.path.join(audio_files_dir, f) for f in os.listdir(audio_files_dir) if f.endswith(".wav")]
    if not audio_files:
        print(f"[ERROR] No .wav files found in {audio_files_dir}")
        sys.exit(1)

    model = load_whisper_model()

    print("[Whisper] Starting transcription...")
    for audio_path in tqdm(audio_files, desc="Transcribing segments"):
        full_transcription, word_segments = transcribe_audio(model, audio_path)
        base_name = os.path.basename(audio_path).replace('.wav', '')
        output_path_txt = os.path.join(output_dir, f"{base_name}_transcript.txt")
        output_path_csv = os.path.join(output_dir, f"{base_name}_words.csv")

        # Save plain text
        with open(output_path_txt, "w") as output_file:
            output_file.write(full_transcription)

        # Save word-level CSV
        if word_segments:
            df_words = pd.DataFrame(word_segments)
            df_words.to_csv(output_path_csv, index=False)

        print(f"[Whisper] Transcription saved to: {output_path_txt}")
        print(f"[Whisper] Word timestamps saved to: {output_path_csv}")

    # Optional: cleanup tmp_wav if created by pipeline
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_dir = os.path.join(project_root, "V0DataSet")
    tmp_wav_path = os.path.join(base_dir, "tmp_wav")
    if os.path.exists(tmp_wav_path):
        shutil.rmtree(tmp_wav_path)
        print(f"[CLEANUP] Deleted full tmp_wav directory: {tmp_wav_path}")

if __name__ == "__main__":
    main()
