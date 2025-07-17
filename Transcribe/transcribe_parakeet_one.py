import os
import sys
import torch
import torchaudio
import pandas as pd
from nemo.collections.asr.models import EncDecCTCModel
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
    waveform, sample_rate = torchaudio.load(audio_path)

    # Check if the audio is stereo (more than one channel)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            mono_audio_path = tmp_file.name
            torchaudio.save(mono_audio_path, waveform, sample_rate)
        path_to_transcribe = mono_audio_path
    else:
        path_to_transcribe = audio_path

    # Perform transcription with word-level timestamps
    result = model.transcribe(
        [path_to_transcribe], 
        return_hypotheses=True, 
        timestamps="word"
    )[0]

    # Remove temporary file if created
    if path_to_transcribe != audio_path:
        os.remove(path_to_transcribe)

    full_text = result.text.strip()
    word_timestamps = result.timestamp['word']

    return full_text, word_timestamps

def transcribe_multiple_audio_files(model, audio_files, output_dir):
    for audio_file in audio_files:
        print(f"[Transcription] Transcribing audio: {audio_file}")
        if not os.path.exists(audio_file):
            print(f"[ERROR] Audio file not found: {audio_file}")
            continue

        text, word_timestamps = transcribe_audio(model, audio_file)

        base_name = os.path.basename(audio_file).replace('.wav', '')
        output_path_txt = os.path.join(output_dir, f"{base_name}_transcript.txt")
        output_path_csv = os.path.join(output_dir, f"{base_name}_words.csv")

        # Save plain text
        with open(output_path_txt, "w") as output_file:
            output_file.write(text)

        # Save word-level CSV
        if word_timestamps:
            df = pd.DataFrame(word_timestamps)
            df.to_csv(output_path_csv, index=False)

        print(f"[Parakeet] Transcription saved to: {output_path_txt}")
        print(f"[Parakeet] Word timestamps saved to: {output_path_csv}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python transcribe_parakeet_one.py <audio_files_dir> <output_dir>")
        sys.exit(1)

    audio_files_dir = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.exists(audio_files_dir):
        print(f"[ERROR] Audio files directory not found: {audio_files_dir}")
        sys.exit(1)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    audio_files = [os.path.join(audio_files_dir, f) for f in os.listdir(audio_files_dir) if f.endswith(".wav")]

    model = load_parakeet_model()
    transcribe_multiple_audio_files(model, audio_files, output_dir)

if __name__ == "__main__":
    main()
