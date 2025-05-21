import os
from transcriptFromAudio import transcriptFromAudio
from addTranscriptToDf import addTranscriptToDf
from addTranscriptToDf import mergeTranscripts
import pandas as pd
output_path = os.path.join(os.path.dirname(__file__), "output")

input_file = os.path.join(os.path.dirname(__file__), "input", "1_video.wav")
input_file_timestamp = os.path.join(os.path.dirname(__file__), "input", "1_segments.csv")

timestamp = []
timestamps_csv = pd.read_csv(input_file_timestamp)["start_time"].to_list()
end_times_csv = pd.read_csv(input_file_timestamp)["end_time"].to_list()

# Combine start and end times into a single list
for start, end in zip(timestamps_csv, end_times_csv):
    if end - start > 3:
        timestamp.extend([int(start), int(end)])
    
        


df_transcript=transcriptFromAudio(audiofile=input_file, outputFolder=output_path, listTimeStamp=timestamp,modelType="tiny.en")

addTranscriptToDf(pd.read_csv(input_file_timestamp),df_transcript)
mergeTranscripts(pd.read_csv(input_file_timestamp),df_transcript)