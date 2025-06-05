import pandas as pd
import os
import subprocess
def splitVideoAndAudioFromSegment(pathVideoMP4, pathSegmentCSV, outputAudioClip, outputVideoClip, segment_frames=64, sliding_window_frames=16, frame_rate=30):

    segments = pd.read_csv(pathSegmentCSV)


    dyadic_segments = []

    videoName = pathVideoMP4.split("/")[-1]

    for _, row in segments.iterrows():
        if row["category"] == "dyadic":
            start_time = row["start_time"]
            end_time = row["end_time"]
            # Ensure the segment duration is at least 5 seconds
            if end_time - start_time < 5:
                continue
            current_start_frame = int(start_time * frame_rate)
            end_frame = int(end_time * frame_rate)

            video_filename = f"{start_time}_to_{end_time}_segment.mp4"
            video_output_path = os.path.join(outputVideoClip, videoName, video_filename)
            os.system(f"ffmpeg -i {pathVideoMP4} -ss {start_time} -to {end_time} -c copy {video_output_path}")

            index=0
            while current_start_frame + segment_frames <= end_frame:
                current_end_frame = current_start_frame + segment_frames
                
                audio_filename = f"{current_start_frame}_to_{current_end_frame}_segment.wav"
                
                # Generate output paths for audio and video
                audio_output_path = os.path.join(outputAudioClip, videoName, audio_filename)
                
                # Create directories for the output paths if they don't exist
                os.makedirs(os.path.dirname(audio_output_path), exist_ok=True)
                os.makedirs(os.path.dirname(video_output_path), exist_ok=True)

                # Use ffmpeg to extract the audio segment in WAV format
                os.system(f"ffmpeg -i {pathVideoMP4} -ss {current_start_frame/30} -to {current_end_frame/30} -q:a 0 -map a -ar 44100 {audio_output_path}")
                
                # Use ffmpeg to extract the video segment
                # subprocess.run([
                #     "ffmpeg", "-i", pathVideoMP4,
                #     "-vf", f"select='between(n,{current_start_frame},{current_end_frame})'",
                #     "-vsync", "vfr", "-c:v", "libx264", "-preset", "ultrafast", video_output_path
                # ], check=True)
                dyadic_segments.append({
                    "start_frame": current_start_frame,
                    "end_frame": current_end_frame,
                })
                
                # Move the sliding window forward
                current_start_frame += sliding_window_frames

    # Save the DataFrame as CSV in the respective directories
    dyadic_segments_df = pd.DataFrame(dyadic_segments)

    audio_csv_path = os.path.join(outputAudioClip, videoName, "dyadic_segments_audio.csv")
    video_csv_path = os.path.join(outputVideoClip, videoName, "dyadic_segments_video.csv")

    dyadic_segments_df.to_csv(audio_csv_path, index=False)
    dyadic_segments_df.to_csv(video_csv_path, index=False)
    # Return a DataFrame containing only dyadic segments
    return pd.DataFrame(dyadic_segments)


if __name__ == "__main__":
    input_path_csv = os.path.join(os.path.dirname(__file__), "../V0.2DataSet/segments/1_segments.csv")
    input_path_mp4 = os.path.join(os.path.dirname(__file__), "../V0.2DataSet/mp4/1_video.mp4")
    output_audio = os.path.join(os.path.dirname(__file__), "../V0.2DataSet/clips_audio/")
    output_video = os.path.join(os.path.dirname(__file__), "../V0.2DataSet/clips_video/")
    print(splitVideoAndAudioFromSegment(input_path_mp4, input_path_csv, output_audio, output_video))
