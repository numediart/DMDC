from nemo.collections.asr.models import SortformerEncLabelModel
import csv
import pandas as pd
import librosa
import soundfile as sf
import os

import os

TEMPFOLDER="./tempsegment"


# load model from Hugging Face model card directly (You need a Hugging Face token)
# diar_model = SortformerEncLabelModel.restore_from(restore_path="./NemoDiarization/model/diar_sortformer_4spk-v1.nemo", map_location='cuda', strict=False)
diar_model = SortformerEncLabelModel.restore_from(restore_path="./NemoDiarization/model/diar_sortformer_4spk-v1.nemo", map_location='cuda', strict=False)




def diarization_nvidia_sortformer_process(audio_input,output_path,segmentation_minutes=5):
    audio_base_name = os.path.basename(audio_input)
    print(f"Base name of the audio input: {audio_base_name}")
    # Create the temporary folder if it doesn't exist
    if not os.path.exists(TEMPFOLDER):
        os.mkdir(TEMPFOLDER)

    # Load the audio file
    audio, sr = librosa.load(audio_input, sr=16000, mono=True)
    segmentation_sec=segmentation_minutes*60


    # Split the audio into n-minute segments
    segment_duration = segmentation_sec * sr  
    segments = [audio[i:i + segment_duration] for i in range(0, len(audio), segment_duration)]

    predicted_segments = []
    for idx, segment in enumerate(segments):
        # Save each segment to a temporary file
        segment_audio_path = f"{TEMPFOLDER}/segment_{idx}_{audio_base_name}.wav"
        sf.write(segment_audio_path, segment, sr)
        
        # Perform diarization on the segment
        segment_predicted = diar_model.diarize(audio=segment_audio_path, batch_size=4,map="cuda", include_tensor_outputs=True)
        
        for diarized_segments in segment_predicted[0]:
            for one_diarized_segment in diarized_segments:
                split_one_diarized_segment = one_diarized_segment.split(" ")
                
                # Create a dictionary for each diarized segment
                segment_predicted_dict = {
                    "start_time": float(split_one_diarized_segment[0]) + (segmentation_sec * idx),
                    "end_time": float(split_one_diarized_segment[1]) + (segmentation_sec * idx),
                    "speaker": split_one_diarized_segment[2],
                }

                predicted_segments.append(segment_predicted_dict)
        

    for file in os.listdir(TEMPFOLDER):
        file_path = os.path.join(TEMPFOLDER, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
    os.rmdir(TEMPFOLDER)


    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(predicted_segments)
    # Order the DataFrame by start_time
    df = df.sort_values(by="start_time").reset_index(drop=True)

    # Save DataFrame to CSV
    output_csv = f"{output_path}/{audio_base_name.replace('.wav', '')}_diarization.csv"
    df.to_csv(output_csv, index=False)

    print(f"Predicted segments saved to {output_csv}")


if __name__ == "__main__":
    audio_input="./NemoDiarization/input/1.1_video.wav"
    output_path="./NemoDiarization/output/"
    diarization_nvidia_sortformer_process(audio_input,output_path, 5)