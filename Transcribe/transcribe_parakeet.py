import os
import sys
import torch
import pandas as pd
from omegaconf import OmegaConf
import nemo.collections.asr as nemo_asr
import shutil
import tempfile
import torchaudio
from pathlib import Path

def load_parakeet_model(model_name="nvidia/parakeet-tdt-0.6b-v2"):
    print("[Parakeet] Loading model:", model_name)
    model = nemo_asr.models.ASRModel.from_pretrained(
        model_name=model_name,
        map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    model.eval()
    return model

def transcribe_audio(model, audio_path, segment_idx=None, word_timestamps_output_dir=None):
    waveform, sample_rate = torchaudio.load(audio_path)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            mono_audio_path = tmp_file.name
            torchaudio.save(mono_audio_path, waveform, sample_rate)
        audio_to_transcribe = mono_audio_path
    else:
        audio_to_transcribe = audio_path

    try:
        result = model.transcribe([audio_to_transcribe], timestamps=True)
        transcription = result[0].text.strip()

        # save timestamps if segment_idx is provided
        if segment_idx is not None and word_timestamps_output_dir is not None:
            word_ts = result[0].timestamp.get("word", [])
            if word_ts:  # list not empty
                Path(word_timestamps_output_dir).mkdir(parents=True, exist_ok=True)
                word_ts_df = pd.DataFrame(word_ts)
                word_ts_df.to_csv(os.path.join(word_timestamps_output_dir, f"segment_{segment_idx}_words.csv"), index=False)
            else:
                print(f"[WARNING] No word-level timestamps for segment {segment_idx}")
        
    except Exception as e:
        print(f"[ERROR] Failed to transcribe {audio_path}: {e}")
        transcription = "[UNTRANSCRIBED]"

    if "mono_audio_path" in locals() and os.path.exists(mono_audio_path):
        os.remove(mono_audio_path)

    return transcription



def main():
    if len(sys.argv) < 3:
        print("Usage: python transcribe_parakeet.py <output_dir> <audio1.wav> <audio2.wav> ...")
        sys.exit(1)

    video_idx = sys.argv[1]
    audio_files = sys.argv[2:]

    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # remonte deux niveaux
    base_dir = os.path.join(project_root, "V0DataSet")
    CUSTOM_SEGMENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test", "clean_segments")

    BASE_DIR = Path(__file__).resolve().parent.parent 
    CLEAN_SEGMENTS_DIR = BASE_DIR / "test" / "clean_segments"
    csv_path = CLEAN_SEGMENTS_DIR / f"{video_idx}_clean_segments.csv"
    WORD_TIMESTAMPS_DIR = BASE_DIR / "V0.10DataSet" / "word" / f"{video_idx}_video"
    WORD_TIMESTAMPS_DIR.mkdir(parents=True, exist_ok=True)

    # Change this line for benchmark :
    segments_csv_path = os.path.join(base_dir, "segments", f"{video_idx}_segments.csv")
    # segments_csv_path = CLEAN_SEGMENTS_DIR / f"{video_idx}_clean_segments.csv"

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
        transcription = transcribe_audio(
            model,
            audio_path,
            segment_idx=i,
            word_timestamps_output_dir=WORD_TIMESTAMPS_DIR
        )
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