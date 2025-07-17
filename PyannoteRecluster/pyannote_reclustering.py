import torch
import torchaudio
import pandas as pd
import numpy as np
from pyannote.audio import Inference
from sklearn.cluster import AgglomerativeClustering
from typing import Optional, Union

# --- GLOBAL CONFIGURATION ---

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBEDDING_MODEL = "pyannote/embedding"

print(f"Loading embedding model '{EMBEDDING_MODEL}' on device '{DEVICE}'...")
try:
    embedding_inference = Inference(EMBEDDING_MODEL, window="whole", device=DEVICE, use_auth_token="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")
except Exception as e:
    print(f"Error while loading Pyannote model: {e}")
    embedding_inference = None

def recluster_pyannote_diarization(
    audio_path: str,
    initial_diarization_csv: str,
    output_csv_path: str,
    num_speakers: int = 2,
    min_segment_duration: float = 0.5
) -> Optional[pd.DataFrame]:

    if embedding_inference is None:
        print("Embedding model could not be loaded. Operation aborted.")
        return None

    print("\n--- STARTING RECLUSTERING ---")
    print(f"Audio file: {audio_path}")
    print(f"Initial diarization: {initial_diarization_csv}")
    print(f"Target number of speakers: {num_speakers}")

    try:
        print("1/5: Loading files...")
        waveform, sample_rate = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        df_diarization = pd.read_csv(initial_diarization_csv)
    except FileNotFoundError as e:
        print(f"ERROR: File not found - {e}")
        return None

    print("2/5: Extracting embeddings per segment...")
    embeddings = []
    valid_segments_indices = []

    for index, row in df_diarization.iterrows():
        start, end = row['start'], row['end']
        if (end - start) < min_segment_duration:
            continue

        # Add a small epsilon to avoid too short segments
        end = max(end, start + 0.01)
        segment_waveform = waveform[:, int(start * sample_rate):int(end * sample_rate)]
        segment_data = {'waveform': segment_waveform, 'sample_rate': sample_rate}

        try:
            embedding = embedding_inference(segment_data)
            embeddings.append(embedding.squeeze())
            valid_segments_indices.append(index)
        except Exception as e:
            print(f"Warning: Embedding error on segment {index}: {e}")

    if not embeddings:
        print("ERROR: No embeddings could be extracted. Check segment durations.")
        return None

    embeddings_matrix = np.array(embeddings)
    print(f"--> {len(embeddings_matrix)} embeddings successfully extracted.")

    print(f"3/5: Clustering into {num_speakers} groups (euclidean/ward)...")
    clustering = AgglomerativeClustering(
        n_clusters=num_speakers,
        metric='euclidean',
        linkage='ward'
    )
    cluster_labels = clustering.fit_predict(embeddings_matrix)

    print("4/5: Reassigning labels and merging segments...")
    df_diarization['speaker'] = "UNKNOWN"
    for i, original_index in enumerate(valid_segments_indices):
        new_speaker_label = f"SPEAKER_{cluster_labels[i]}"
        df_diarization.loc[original_index, 'speaker'] = new_speaker_label

    # Fill unknown values using forward and backward fill
    df_diarization['speaker'] = df_diarization['speaker'].replace('UNKNOWN', method='ffill').replace('UNKNOWN', method='bfill')

    def _merge_consecutive_segments(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty: return df
        df = df.sort_values(by='start').reset_index(drop=True)
        merged_rows = []
        current_row = df.iloc[0].copy()
        for i in range(1, len(df)):
            next_row = df.iloc[i]
            # Merge if same speaker and segments are close
            if next_row['speaker'] == current_row['speaker'] and next_row['start'] - current_row['end'] < 0.1:
                current_row['end'] = max(current_row['end'], next_row['end'])
            else:
                merged_rows.append(current_row)
                current_row = next_row.copy()
        merged_rows.append(current_row)
        return pd.DataFrame(merged_rows)

    df_merged = _merge_consecutive_segments(df_diarization)
    df_final = df_merged[['start', 'end', 'speaker']].copy()

    print(f"5/5: Saving result to {output_csv_path}...")
    df_final.to_csv(output_csv_path, index=False)

    print(f"✅ Reclustering completed successfully.")
    return df_final

if __name__ == '__main__':
    print("--- STARTING EXAMPLE RUN ---")

    for i in range(1, 11):
        AUDIO_FILE = f"V0DataSet/wav/{i}_video.wav"
        INITIAL_DIARIZATION_CSV = f"V0DataSet/Diarization_Results/{i}_video_diarization_results.csv"
        FINAL_DIARIZATION_CSV = f"V0DataSet/Diarization_Results/{i}_video_diarization_reclustered.csv"
        NUM_SPEAKERS = 2

        print(f"\nProcessing file: {AUDIO_FILE}")
        final_df = recluster_pyannote_diarization(
            audio_path=AUDIO_FILE,
            initial_diarization_csv=INITIAL_DIARIZATION_CSV,
            output_csv_path=FINAL_DIARIZATION_CSV,
            num_speakers=NUM_SPEAKERS
        )

        if final_df is not None:
            print("Preview of final DataFrame returned by the function:")
            print(final_df.head())
        else:
            print("Error while processing this file.")
