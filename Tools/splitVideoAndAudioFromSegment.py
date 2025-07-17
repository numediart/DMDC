import pandas as pd
import os


def extract_audio_segment(input_wav, start_frame, end_frame, output_wav, frame_rate=30):
    # Ensure output_wav is a file path with .wav extension
    base_name = os.path.splitext(os.path.basename(input_wav))[0]
    output_wav = os.path.join(output_wav, f"{base_name}_{start_frame}_to_{end_frame}.wav")
    # Calculate start and end times in seconds
    start_time = start_frame / frame_rate
    duration = (end_frame - start_frame) / frame_rate
    # Use ffmpeg to extract the segment
    cmd = f'ffmpeg -loglevel error -y -ss {start_time} -i "{input_wav}" -t {duration} -c copy "{output_wav}"'
    # cmd = f'ffmpeg -y -ss {start_time} -i "{input_wav}" -t {duration} -acodec pcm_s16le -ar 16000 "{output_wav}"'
    os.system(cmd)
    

def extract_video_segments(pathVideoMP4, pathSegmentCSV, outputVideoClip, frame_rate=30):
    segments = pd.read_csv(pathSegmentCSV)
    videoName = os.path.basename(pathVideoMP4)
    dyadic_segments = []

    for _, row in segments.iterrows():
        if row["category"] == "dyadic":
            start_time = row["start_time"]
            end_time = row["end_time"]
            if end_time - start_time <= 5:
                continue
            video_filename = f"{start_time}_to_{end_time}_segment.mp4"
            video_output_path = os.path.join(outputVideoClip, videoName, video_filename)
            os.makedirs(os.path.dirname(video_output_path), exist_ok=True)
            os.system(f"ffmpeg -ss {start_time} -i {pathVideoMP4} -t {end_time - start_time} -c copy {video_output_path}")
            dyadic_segments.append({
                "start_time": start_time,
                "end_time": end_time,
                "speaker": row["speaker"]
            })

    dyadic_segments_df = pd.DataFrame(dyadic_segments)
    video_csv_path = os.path.join(outputVideoClip, videoName, "dyadic_segments_video.csv")
    dyadic_segments_df.to_csv(video_csv_path, index=False)
    return dyadic_segments_df

if __name__ == "__main__":
    input_path_csv = os.path.join(os.path.dirname(__file__), "../V0.2DataSet/segments/1_segments.csv")
    input_path_mp4 = os.path.join(os.path.dirname(__file__), "../V0.2DataSet/mp4/1_video.mp4")
    output_audio = os.path.join(os.path.dirname(__file__), "../V0.2DataSet/clips_audio/")
    output_video = os.path.join(os.path.dirname(__file__), "../V0.2DataSet/clips_video/")
    print(extract_audio_segments(input_path_mp4, input_path_csv, output_audio))
    print(extract_video_segments(input_path_mp4, input_path_csv, output_video))
