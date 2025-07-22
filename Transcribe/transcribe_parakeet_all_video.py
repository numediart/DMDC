import os
import sys
import torch
import torchaudio
from nemo.collections.asr.models import EncDecCTCModel
import tempfile

SEGMENT_LENGTH = 120  # seconds

def load_parakeet_model(model_name="nvidia/parakeet-tdt-0.6b-v2"):
    print("[Parakeet] Loading model:", model_name)
    model = EncDecCTCModel.from_pretrained(
        model_name=model_name, 
        map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    model.eval()
    return model

def segment_audio(waveform, sample_rate, segment_length=SEGMENT_LENGTH):
    total_samples = waveform.shape[1]
    segment_samples = int(segment_length * sample_rate)
    segments = []
    for start in range(0, total_samples, segment_samples):
        end = min(start + segment_samples, total_samples)
        segments.append(waveform[:, start:end])
    return segments

def transcribe_audio_segments(model, audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    # Convert to mono if needed
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    segments = segment_audio(waveform, sample_rate)
    full_text = ""

    for i, segment in enumerate(segments):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            segment_path = tmp_file.name
            torchaudio.save(segment_path, segment, sample_rate)
        result = model.transcribe([segment_path])[0]
        os.remove(segment_path)
        segment_text = result.strip()
        full_text += (" " if full_text else "") + segment_text

    return full_text

def main():
    if len(sys.argv) < 3:
        print("Usage: python transcribe_parakeet_all_video.py <audio_file.wav> <output_dir>")
        sys.exit(1)

    audio_file = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.exists(audio_file):
        print(f"[ERROR] Audio file not found: {audio_file}")
        sys.exit(1)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model = load_parakeet_model()
    print(f"[Transcription] Transcribing audio: {audio_file}")
    text = transcribe_audio_segments(model, audio_file)

    base_name = os.path.basename(audio_file).replace('.wav', '')
    output_path_txt = os.path.join(output_dir, f"{base_name}_transcript.txt")

    with open(output_path_txt, "w") as output_file:
        output_file.write(text)

    print(f"[Parakeet] Transcription saved to: {output_path_txt}")

if __name__ == "__main__":
    main()
