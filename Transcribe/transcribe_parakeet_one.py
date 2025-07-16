import os
import sys
import torch
import torchaudio
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
        # Convert to mono by averaging the channels
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Create a temporary file for the mono version
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            mono_audio_path = tmp_file.name
            torchaudio.save(mono_audio_path, waveform, sample_rate)
        
        # Transcribe the mono file
        result = model.transcribe([mono_audio_path])
        
        # Delete the temporary file
        os.remove(mono_audio_path)
    else:
        # If already mono, transcribe directly
        result = model.transcribe([audio_path])

    return result[0].text.strip()

def transcribe_multiple_audio_files(model, audio_files, output_dir):
    for audio_file in audio_files:
        print(f"[Transcription] Transcribing audio: {audio_file}")
        if not os.path.exists(audio_file):
            print(f"[ERROR] Audio file not found: {audio_file}")
            continue

        transcription = transcribe_audio(model, audio_file)
        output_path_txt = os.path.join(output_dir, f"{os.path.basename(audio_file).replace('.wav', '_transcript.txt')}")

        with open(output_path_txt, "w") as output_file:
            output_file.write(transcription)

        print(f"[Parakeet] Transcription saved to: {output_path_txt}")

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
