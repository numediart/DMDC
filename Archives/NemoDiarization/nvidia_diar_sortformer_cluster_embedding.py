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




def diarization_nvidia_sortformer_process(audio_input, output_path, audio_divide_ratio=5, disable_segmentation=False):
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

    # Calculate the duration of each segment based on the audio_divide_ratio
    total_samples = len(audio)
    video_lengh=total_samples/sr
    print("video_lenght",video_lengh)
    segment_duration = total_samples // audio_divide_ratio
    
    # Split the audio into equal parts based on audio_divide_ratio
    segments = [audio[i:i + segment_duration] for i in range(0, total_samples, segment_duration)]

    # Ensure the last segment is included if the division isn't perfect
    if len(segments) > audio_divide_ratio:
        segments[-2] = np.concatenate([segments[-2], segments[-1]])
        segments = segments[:-1]

    all_embeddings=[]
    segments_data = []
    index_embedding=0
    for idx, segment in enumerate(segments):
        # Save each segment to a temporary file
        segment_audio_path = f"{TEMPFOLDER}/segment_{idx}_{audio_base_name}.wav"
        sf.write(segment_audio_path, segment, sr)


        predicted_segments, predicted_probs = diar_model.diarize(audio=segment_audio_path, batch_size=1, include_tensor_outputs=True)
        
        embeddings = predicted_probs[0]
        for embedding in embeddings:
            all_embeddings.append(embedding)

        diarized_segments = predicted_segments[0]
        for one_diarized_segment, embedding in zip(diarized_segments, embeddings):
            split_one_diarized_segment = one_diarized_segment.split(" ")

            segment_predicted_dict = {
            "start_time": float(split_one_diarized_segment[0]) + (segment_duration / sr * idx),
            "end_time": float(split_one_diarized_segment[1]) + (segment_duration / sr * idx),
            "speaker": split_one_diarized_segment[2][:-1].upper() + "0" + split_one_diarized_segment[2][-1].upper(),
            # "embedding": embedding.tolist(),  # Add embedding to the dictionary
            "embedding_index_start":index_embedding,
            "embedding_index_end":index_embedding+len(embedding),
            }
            index_embedding+=len(embedding)

            segments_data.append(segment_predicted_dict)


    # Convert segment data to a DataFrame
    df = pd.DataFrame(segments_data)
    print(df)
    embedding_output_path = os.path.join(output_path, f"embedding_results_{audio_base_name}.csv")
    df.to_csv(embedding_output_path)

    # Perform clustering using KMeans on the embeddings
    num_speakers=2

    # Not Sure
    flattened_embeddings = np.vstack([embedding.numpy() for embedding in all_embeddings])

    kmeans = KMeans(n_clusters=num_speakers, random_state=0)

    kmeans.fit(flattened_embeddings)

    # Assign cluster labels to embeddings
    cluster_labels = kmeans.labels_

    # Map cluster labels to speaker IDs
    speaker_mapping = {i: f"SPEAKER_0{i}" for i in range(num_speakers)}

    # Create a DataFrame for clustering results
    clustering_results = []
    for idx, label in enumerate(cluster_labels):
        clustering_results.append({
            "embedding_index": idx,
            "start_time": round(idx/12.5,3),
            "end_time": round(idx/12.5,3)+0.08,
            "speaker": speaker_mapping[label]
        })

    clustering_df = pd.DataFrame(clustering_results)
    print(clustering_df)

    # Save clustering results to a CSV file
    clustering_output_path = os.path.join(output_path, f"clustering_results_{audio_base_name}.csv")
    clustering_df.to_csv(clustering_output_path, index=False)
    print(f"Clustering results saved to {clustering_output_path}")


if __name__ == "__main__":
    output_path="./NemoDiarization/output/embedding/"
    for i in range(1,2):
        audio_input="./V0.2DataSet/wav/"+str(i)+"_video"
        diarization_nvidia_sortformer_process(audio_input,output_path,audio_divide_ratio=3)
    # audio_input="./NemoDiarization/input/1.1_video.wav"
    # diarization_nvidia_sortformer_process(audio_input,output_path, 5)