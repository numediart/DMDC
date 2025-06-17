from nemo.collections.asr.models import SortformerEncLabelModel
import csv
import pandas as pd
import librosa
import soundfile as sf
import os

import os

TEMPFOLDER="./tempsegment"


diar_model = SortformerEncLabelModel.restore_from(restore_path="./NemoDiarization/model/diar_sortformer_4spk-v1.nemo", map_location='cuda', strict=False)




def diarization_nvidia_sortformer_process(audio_input, output_path, segmentation_minutes=5,overlap_duration_sec=60):
    """
    Perform speaker diarization on an audio file using NVIDIA's SortFormer model.
    Args:
        audio_input (str): Path to the input audio file.
        output_path (str): Directory to save the diarization results as a CSV file.
        segmentation_minutes (int, optional): Duration of audio segments in minutes. Defaults to 5.
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

    predicted_segments = []

   
    segment_duration = segmentation_sec * sr
    overlap_samples = int(overlap_duration_sec * sr)
    step_size = segment_duration - overlap_samples
    segments = [
        audio[i:min(len(audio), i + segment_duration)]
        for i in range(0, len(audio), step_size)
    ]

    for idx, segment in enumerate(segments):
        # Save each segment to a temporary file
        segment_audio_path = f"{TEMPFOLDER}/segment_{idx}_{audio_base_name}.wav"
        sf.write(segment_audio_path, segment, sr)

        segment_predicted = diar_model.diarize(audio=segment_audio_path, batch_size=1)
        # For each segment, extract information to fit our requirements (start time, end time, speaker)
        for one_diarized_segment in segment_predicted[0]:
            split_one_diarized_segment = one_diarized_segment.split(" ")

            segment_predicted_dict = {
                "start_time": round(float(split_one_diarized_segment[0]) + (segmentation_sec * idx)-(overlap_duration_sec*idx),2),
                "end_time": round(float(split_one_diarized_segment[1]) + (segmentation_sec * idx)-(overlap_duration_sec*idx),2),
                "speaker": split_one_diarized_segment[2][:-1].upper() + "0" + split_one_diarized_segment[2][-1].upper(),
                "duration": round(float(split_one_diarized_segment[1]) - float(split_one_diarized_segment[0]),2),
                "segment_number":idx,
            }

            predicted_segments.append(segment_predicted_dict)

    # Clean up temporary files
    for file in os.listdir(TEMPFOLDER):
        file_path = os.path.join(TEMPFOLDER, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
    os.rmdir(TEMPFOLDER)

    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(predicted_segments)
    # Order the DataFrame by start_time
    df = df.sort_values(by="start_time").reset_index(drop=True)

    # Remove overlapping segments and identify primary speakers
    cleaned_segments = []
    for idx, segment in enumerate(predicted_segments):
        if idx == 0 or segment["start_time"] >= predicted_segments[idx - 1]["end_time"]:
            cleaned_segments.append(segment)
        else:
            # Handle overlap by assigning the speaker with the longest duration
            prev_segment = predicted_segments[idx - 1]
            overlap_duration = prev_segment["end_time"] - segment["start_time"]
            if overlap_duration > 0:
                if prev_segment["duration"] >= segment["duration"]:
                    segment["speaker"] = prev_segment["speaker"]
                else:
                    prev_segment["speaker"] = segment["speaker"]

    # Convert the cleaned list of dictionaries into a DataFrame
    df = pd.DataFrame(cleaned_segments)

    # Save DataFrame to CSV
    output_csv = f"{output_path}/{audio_base_name.replace('.wav', '')}_diarization.csv"
    df.to_csv(output_csv, index=False)

    print(f"Predicted segments saved to {output_csv}")


if __name__ == "__main__":
    output_path="./NemoDiarization/output/v0/"
    # for i in range(1,2):
    #     audio_input="./V0.2DataSet/wav/"+str(i)+"_video"
    #     diarization_nvidia_sortformer_process(audio_input,output_path)
    audio_input="./V0.2DataSet/wav/"+str(1)+"_video"
    diarization_nvidia_sortformer_process(audio_input,output_path)