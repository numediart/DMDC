import os
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.cluster import KMeans
from pyannote.audio.pipelines import VoiceActivityDetection
import nemo.collections.asr as nemo_asr

def titanet_diarization_to_csv(
    audio_path,
    output_csv,
    n_speakers=2,
    vad_model_name="pyannote/voice-activity-detection",
    speaker_model_name="nvidia/speakerverification_en_titanet_large",
    vad_token=None,
):
    """
    Diarize an audio file using TitaNet and save results as a CSV.
    # Pipeline schema :
    #
    #  Audio File
    #      │
    #      ▼
    #  ┌───────────────┐
    #  │ Voice Activity│
    #  │  Detection    │
    #  └───────────────┘
    #      │
    #      ▼
    #  ┌───────────────┐
    #  │ Segment Audio │
    #  └───────────────┘
    #      │
    #      ▼
    #  ┌───────────────┐
    #  │ Speaker       │
    #  │ Embedding     │
    #  └───────────────┘
    #      │
    #      ▼
    #  ┌───────────────┐
    #  │ Clustering    │
    #  │ (KMeans)      │
    #  └───────────────┘
    #      │
    #      ▼
    #  ┌───────────────┐
    #  │ CSV Output    │
    #  └───────────────┘
    Args:
        audio_path (str): Path to input audio file.
        output_csv (str): Path to output CSV file.
        n_speakers (int): Number of speakers to cluster.
        vad_model_name (str): HuggingFace model name for VAD.
        speaker_model_name (str): NeMo model name for speaker embedding.
        vad_token (str): HuggingFace token for VAD model.
    """
    # Load models
    speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(speaker_model_name)
    vad_pipeline = VoiceActivityDetection.from_pretrained(vad_model_name, use_auth_token=vad_token)

    # Load audio
    audio, sr = sf.read(audio_path)

    # Get speech segments
    segmentation = vad_pipeline(audio_path)
    segments = []
    segment_times = []
    for speech_turn in segmentation.get_timeline():
        start = speech_turn.start
        end = speech_turn.end
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        segments.append(audio[start_sample:end_sample])
        segment_times.append((start, end))

    # Extract embeddings
    embeddings = []
    for idx, segment in enumerate(segments):
        temp_path = f"temp_segment_{idx}.wav"
        sf.write(temp_path, segment, sr)
        emb = speaker_model.get_embedding(temp_path)
        embeddings.append(emb.cpu().numpy().flatten())
        os.remove(temp_path)
    embeddings = np.stack(embeddings)

    # Cluster
    kmeans = KMeans(n_clusters=n_speakers, random_state=0)
    labels = kmeans.fit_predict(embeddings)

    # DataFrame
    diar_results = []
    for (start, end), label in zip(segment_times, labels):
        diar_results.append({
            "start_time": round(start, 3),
            "end_time": round(end, 3),
            "speaker": f"SPEAKER_{label+1}"
        })
    df = pd.DataFrame(diar_results)
    df = df.sort_values(by="start_time").reset_index(drop=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved diarization CSV to {output_csv}")


if __name__ == "__main__":
    # Set your HuggingFace token if needed for pyannote
    HF_TOKEN = "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"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(base_dir, "input")
    output_folder = os.path.join(base_dir, "output")
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".wav"):
            audio_path = os.path.join(input_folder, filename)
            output_csv = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_diarization.csv")
            titanet_diarization_to_csv(audio_path, output_csv, n_speakers=2, vad_token=HF_TOKEN)
    titanet_diarization_to_csv(audio_path, output_csv, n_speakers=2, vad_token=HF_TOKEN)
