from nemo.collections.asr.models import SortformerEncLabelModel
import csv
import pandas as pd
import librosa
import soundfile as sf
import os

import os

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

    predicted_segments = []

    if disable_segmentation:
        # Process the entire audio file without segmentation
        segment_audio_path = f"{TEMPFOLDER}/full_audio_{audio_base_name}.wav"
        sf.write(segment_audio_path, audio, sr)

        segment_predicted = diar_model.diarize(audio=segment_audio_path, batch_size=1)
        for diarized_segments in segment_predicted[0]:
            for one_diarized_segment in diarized_segments:
                split_one_diarized_segment = one_diarized_segment.split(" ")

                segment_predicted_dict = {
                    "start_time": float(split_one_diarized_segment[0]),
                    "end_time": float(split_one_diarized_segment[1]),
                    "speaker": split_one_diarized_segment[2][:-1].upper() + "0" + split_one_diarized_segment[2][-1].upper(),
                }

                predicted_segments.append(segment_predicted_dict)
    else:
        # Split the audio into n-minute segments
        segment_duration = segmentation_sec * sr
        segments = [audio[i:i + segment_duration] for i in range(0, len(audio), segment_duration)]

        for idx, segment in enumerate(segments):
            # Save each segment to a temporary file
            segment_audio_path = f"{TEMPFOLDER}/segment_{idx}_{audio_base_name}.wav"
            sf.write(segment_audio_path, segment, sr)

            segment_predicted = diar_model.diarize(audio=segment_audio_path, batch_size=4, include_tensor_outputs=True)
            # For each segment, extract information to fit our requirements (start time, end time, speaker)
            for diarized_segments in segment_predicted[0]:
                for one_diarized_segment in diarized_segments:
                    split_one_diarized_segment = one_diarized_segment.split(" ")

                    segment_predicted_dict = {
                        "start_time": float(split_one_diarized_segment[0]) + (segmentation_sec * idx),
                        "end_time": float(split_one_diarized_segment[1]) + (segmentation_sec * idx),
                        "speaker": split_one_diarized_segment[2][:-1].upper() + "0" + split_one_diarized_segment[2][-1].upper(),
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

    # Save DataFrame to CSV
    output_csv = f"{output_path}/{audio_base_name.replace('.wav', '')}_diarization.csv"
    df.to_csv(output_csv, index=False)

    print(f"Predicted segments saved to {output_csv}")


if __name__ == "__main__":
    output_path="./NemoDiarization/output/v0/"
    for i in range(1,11):
        audio_input="./V0.2DataSet/wav/"+str(i)+"_video"
        diarization_nvidia_sortformer_process(audio_input,output_path)
    # audio_input="./NemoDiarization/input/thedeepskintest.wav"
    # diarization_nvidia_sortformer_process(audio_input,output_path, 5,disable_segmentation=True)