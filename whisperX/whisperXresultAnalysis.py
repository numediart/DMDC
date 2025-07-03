import pandas as pd
import os


import json
def process_diarization_data(input_json_path, output_csv_path):
    """
    Processes diarization data from a JSON file, extracts word-level information,
    and groups speaker segments based on speaking times, saving the result in a single CSV file.

    Args:
        input_json_path (str): Path to the input JSON file.
        output_csv_path (str): Path to save the grouped speaker segments CSV data.
    """
    # Load JSON data
    with open(input_json_path, 'r') as file:
        json_data = [json.loads(line) for line in file]
    diarization_whisperX = pd.DataFrame(json_data)

    # Extract word-level information
    diarization_wordperword_whisperX = diarization_whisperX["words"]
    all_words = [
        word_dict
        for segment in diarization_wordperword_whisperX
        for word_dict in segment
    ]

    # Create a DataFrame for all words
    df_all_words = pd.DataFrame(all_words)

    # Regroup speaker segments based on speaking times
    segments = []
    current_segment = {"start_time": None, "end_time": None, "speaker": None}

    for _, row in df_all_words.iterrows():
        if current_segment["speaker"] is None or current_segment["speaker"] != row["speaker"]:
            # Save the current segment if it exists
            if current_segment["speaker"] is not None:
                segments.append(current_segment)
            # Start a new segment
            current_segment = {
                "start_time": row["start"],
                "end_time": row["end"],
                "speaker": row["speaker"],
            }
        else:
            # Extend the current segment
            current_segment["end_time"] = row["end"]

    # Add the last segment
    if current_segment["speaker"] is not None:
        segments.append(current_segment)

    # Create a DataFrame for the segments
    df_segments = pd.DataFrame(segments)

    # Save grouped speaker segments to CSV
    df_segments.to_csv(output_csv_path, index=False)


# Example usage
if __name__ == "__main__":
    input_path = os.path.join(os.path.dirname(__file__), "output/7_video_diarization.json")
    output_csv = os.path.join(os.path.dirname(__file__), "output/7_video_diarization.csv")

    process_diarization_data(input_path, output_csv)
