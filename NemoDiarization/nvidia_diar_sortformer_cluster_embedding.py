from nemo.collections.asr.models import SortformerEncLabelModel
import csv
import pandas as pd
import librosa
import soundfile as sf
import os
import numpy as np
import os
from sklearn.cluster import KMeans

TEMPFOLDER="./tempsegment"


diar_model = SortformerEncLabelModel.restore_from(restore_path="./NemoDiarization/model/diar_sortformer_4spk-v1.nemo", map_location='cuda', strict=False)




def diarization_nvidia_sortformer_process(audio_input, output_path, segmentation_minutes=5, disable_segmentation=False):
    """
    Perform speaker diarization on an audio file using NVIDIA's SortFormer model.
    Args:
        audio_input (str): Path to the input audio file.
        output_path (str): Directory to save the diarization results as a CSV file.
        segmentation_minutes (int, optional): Duration of audio segments in minutes. Defaults to 5.
        disable_segmentation (bool, optional): If True, disables segmentation and processes the entire audio file. Defaults to False.
    Returns:
        None: Saves the diarization results to a CSV file in the specified output directory.
    """

    audio_base_name = os.path.basename(audio_input)
    print(f"Base name of the audio input: {audio_base_name}")

    # Create the temporary folder if it doesn't exist
    if not os.path.exists(TEMPFOLDER):
        os.mkdir(TEMPFOLDER)

    # Load the audio file
    audio, sr = librosa.load(audio_input, sr=16000, mono=True)
    segmentation_sec = segmentation_minutes * 60



    # Split the audio into n-minute segments
    segment_duration = segmentation_sec * sr
    segments = [audio[i:i + segment_duration] for i in range(0, len(audio), segment_duration)]

    all_embeddings=[]

    for idx, segment in enumerate(segments):
        # Save each segment to a temporary file
        segment_audio_path = f"{TEMPFOLDER}/segment_{idx}_{audio_base_name}.wav"
        sf.write(segment_audio_path, segment, sr)


        predicted_segments, predicted_probs = diar_model.diarize(audio=segment_audio_path, batch_size=1, include_tensor_outputs=True)
        
        embeddings = predicted_probs[0]
        for embedding in embeddings:
            all_embeddings.append(embedding)

        diarized_segments = predicted_segments[0]
        segment_data = []
        for one_diarized_segment in diarized_segments:
            split_one_diarized_segment = one_diarized_segment.split(" ")

            segment_predicted_dict = {
            "start_time": float(split_one_diarized_segment[0]) + (segmentation_sec * idx),
            "end_time": float(split_one_diarized_segment[1]) + (segmentation_sec * idx),
            "speaker": split_one_diarized_segment[2][:-1].upper() + "0" + split_one_diarized_segment[2][-1].upper(),
            }

            segment_data.append(segment_predicted_dict)

        # Convert segment data to a DataFrame
        df = pd.DataFrame(segment_data)
        print(df)

    
    # Perform clustering using KMeans on the embeddings
    num_speakers = 2  
    kmeans = KMeans(n_clusters=num_speakers, random_state=0)
    kmeans.fit(all_embeddings)

    # Assign cluster labels to embeddings
    cluster_labels = kmeans.labels_

    # Map cluster labels to speaker IDs
    speaker_mapping = {i: f"SPEAKER_{i}" for i in range(num_speakers)}

    # Create a DataFrame for clustering results
    clustering_results = []
    for idx, label in enumerate(cluster_labels):
        clustering_results.append({
            "embedding_index": idx,
            "speaker": speaker_mapping[label]
        })

    clustering_df = pd.DataFrame(clustering_results)
    print(clustering_df)

    # Save clustering results to a CSV file
    clustering_output_path = os.path.join(output_path, f"clustering_results_{audio_base_name}.csv")
    clustering_df.to_csv(clustering_output_path, index=False)
    print(f"Clustering results saved to {clustering_output_path}")


if __name__ == "__main__":
    output_path="./NemoDiarization/output/v0/"
    for i in range(1,2):
        audio_input="./V0.2DataSet/wav/"+str(i)+"_video"
        diarization_nvidia_sortformer_process(audio_input,output_path)
    # audio_input="./NemoDiarization/input/1.1_video.wav"
    # diarization_nvidia_sortformer_process(audio_input,output_path, 5)