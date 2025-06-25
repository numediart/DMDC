import pandas as pd
import os


import json

input = os.path.join(os.path.dirname(__file__), "output/7_video_diarization.json")
with open(input, 'r') as file:
    json_data = [json.loads(line) for line in file]
diarization_whisperX = pd.DataFrame(json_data)




diarization_wordperword_whisperX=diarization_whisperX["words"]

allword=[]
for one_segment in diarization_wordperword_whisperX:
    # print(one_segment)
    for one_word_dict in one_segment:
        # print(one_word_dict)
        allword.append(one_word_dict)
        

df_allword=pd.DataFrame(allword)
print(df_allword.head())

output_path = os.path.join(os.path.dirname(__file__), "output/all_words.json")
df_allword.to_json(output_path, orient="records", lines=True)


# Regroup speaker segments based on when they are speaking
segments = []
current_segment = {
    "start_time": None,
    "end_time": None,
    "speaker": None
}

for _, row in df_allword.iterrows():
    if current_segment["speaker"] is None or current_segment["speaker"] != row["speaker"]:
        # Save the current segment if it exists
        if current_segment["speaker"] is not None:
            segments.append(current_segment)
        # Start a new segment
        current_segment = {
            "start_time": row["start"],
            "end_time": row["end"],
            "speaker": row["speaker"]
        }
    else:
        # Extend the current segment
        current_segment["end_time"] = row["end"]

# Add the last segment
if current_segment["speaker"] is not None:
    segments.append(current_segment)

# Create a DataFrame for the segments
df_segments = pd.DataFrame(segments)
print(df_segments.head())

output_csv_path = os.path.join(os.path.dirname(__file__), "output/7_video_diarization.csv")
df_segments.to_csv(output_csv_path, index=False)
