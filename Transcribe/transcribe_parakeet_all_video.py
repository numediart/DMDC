import os
import sys
import torch
import torchaudio
from nemo.collections.asr.models import EncDecCTCModel
import tempfile

SEGMENT_LENGTH = 60  # seconds

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

def transcribe_audio_segments(model, audio_path, segment_length=SEGMENT_LENGTH):
    waveform, sample_rate = torchaudio.load(audio_path)
    # Convert to mono if needed
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    segments = segment_audio(waveform, sample_rate, segment_length)
    results = []  # List of (start_time, end_time, transcript)

    for i, segment in enumerate(segments):
        start_time = i * segment_length
        end_time = min((i + 1) * segment_length, waveform.shape[1] / sample_rate)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            segment_path = tmp_file.name
            torchaudio.save(segment_path, segment, sample_rate)
        result = model.transcribe([segment_path])[0]
        os.remove(segment_path)
        segment_text = result.text.strip()  # Extract text from Hypothesis object
        results.append((start_time, end_time, segment_text))

    return results


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
    segment_results = transcribe_audio_segments(model, audio_file)

    base_name = os.path.basename(audio_file).replace('.wav', '')
    output_path_tsv = os.path.join(output_dir, f"{base_name}_transcript.tsv")

    with open(output_path_tsv, "w", encoding="utf-8") as output_file:
        output_file.write("start_time\tend_time\ttranscript\n")
        for start_time, end_time, transcript in segment_results:
            output_file.write(f"{start_time:.2f}\t{end_time:.2f}\t{transcript}\n")

    print(f"[Parakeet] Transcription saved to: {output_path_tsv}")

if __name__ == "__main__":
    main()
