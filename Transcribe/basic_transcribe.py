import whisper

def load_whisper_model(model_type="LARGE-V2n"):
    """Load a Whisper model."""
    return whisper.load_model(model_type)

def transcribe_audio_with_timestamps(model, audio_path):
    """Transcribe an audio file and return segments with timestamps."""
    result = model.transcribe(audio_path, word_timestamps=False)
    return result["segments"]

def save_segments_to_tsv(segments, tsv_path):
    """Save transcription segments with timestamps to a TSV file."""
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("start\tend\ttext\n")
        for seg in segments:
            start = seg["start"]
            end = seg["end"]
            text = seg["text"].strip().replace("\t", " ")
            f.write(f"{start:.2f}\t{end:.2f}\t{text}\n")

# Example usage:
if __name__ == "__main__":
    model = load_whisper_model("medium.en")
    audio_path = "./Transcribe/14_video.wav"  # Replace with your audio file path
    segments = transcribe_audio_with_timestamps(model, audio_path)
    tsv_path = "14_transcribe.tsv"
    save_segments_to_tsv(segments, tsv_path)
    print(f"Transcription with timestamps saved to {tsv_path}")