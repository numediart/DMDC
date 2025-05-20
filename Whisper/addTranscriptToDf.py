import pandas as pd 
import os

def addTranscriptToDf(df_segment,df_transcript_associated):
    df_segment["transcript"] = ""
    
    # print(df)
    for indexEndSegmentTime in range(len(df_segment["end_time"])):
        for indexEndTranscriptTime in range(len(df_transcript_associated["end"])):

            endSegmentTime=df_segment["end_time"][indexEndSegmentTime]
            endTranscriptTime=df_transcript_associated["end"][indexEndTranscriptTime]

            startSegmentTime=df_segment["start_time"][indexEndSegmentTime]
            startTranscriptTime=df_transcript_associated["start"][indexEndTranscriptTime]

            margin = 0.1  
            if (endSegmentTime * (1 + margin) > endTranscriptTime and startSegmentTime * (1 - margin) < startTranscriptTime):
                df_segment.at[indexEndSegmentTime, "transcript"] = df_segment["transcript"][indexEndSegmentTime] + df_transcript_associated["text"][indexEndTranscriptTime]

    print(df_segment)
    df_segment.to_csv("example.csv", index=False)


current_dir = os.path.dirname(__file__)
df_path = os.path.join(current_dir, "../V0DataSet/segments/1_segments.csv")
transcript_path = os.path.join(current_dir, "../V0DataSet/transcript/1_video/1_video_transcript.csv")
addTranscriptToDf(pd.read_csv(df_path),pd.read_csv(transcript_path))