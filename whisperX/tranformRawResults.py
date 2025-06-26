import pandas as pd
import json
import os

def json_to_diarization(json_file_path, diarization_file_path):
    # Load JSON data from file line by line
    with open(json_file_path, 'r') as file:
        json_data = [json.loads(line) for line in file]

    # Convert JSON data to a pandas DataFrame
    diarization_whisperX = pd.DataFrame(json_data)

    # Rename columns to match diarization format
    diarization_whisperX = diarization_whisperX.rename(columns={'start': 'start_time', 'end': 'end_time', 'speaker': 'speaker'})

    # Export to diarization format (CSV)
    diarization_whisperX.to_csv(diarization_file_path, index=False)

# Example usage
# Get the current working directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define relative paths
json_file_path = os.path.join(current_dir, "input", '7_video_diarization.json')  # Replace with your JSON file path
diarization_file_path = os.path.join(current_dir,"output", '7_video_diarization_raw.csv')  # Replace with your desired diarization file path
json_to_diarization(json_file_path, diarization_file_path)
