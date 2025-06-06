import cv2
import os
from OpenFace.actionUnitExtract import process_FaceLandMark_from_container
from OpenFace.actionUnitExtractVideo import process_FaceLandmarkVidMulti_from_container
import shutil
import time
import csv
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import subprocess
import pandas as pd

TEMP_CLIPS_DIR = "V0DataSet/temp_clips"

# start_time_all_pross=time.time()

def process_FaceLandMark_video(video_path, output,tempfolder = "temp_frames", seconds=1):
    """
    Extracts frames from a video.
    Args:
        video_path (str): Path to the input video file.
        output (str):  Path to the output process file.
        tempFolder(str,optional): Path where the frames will me temporary put.
        seconds (int, optional): Number of seconds to process. Defaults to 1.
    """

    # Create output folder if it doesn't exist
    os.makedirs(tempfolder, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the frame rate of the video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0

    # Process the first 'seconds' of the video
    while frame_count < fps * seconds:
        ret, frame = cap.read()
        if not ret:
            break
        # Save the frame as an image
        frame_filename = os.path.join(tempfolder, f'frame_{frame_count:04d}.jpg')
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()
    print(f"[AU/PreProcess] Frames saved in '{tempfolder}'")

    # Process all frames
    for frame_file in sorted(os.listdir(tempfolder)):
        if frame_file.endswith('.jpg'):
            start_time = time.time()
            input_path = os.path.join(tempfolder, frame_file)
            print("----------------------------------------------------")
            print("[AU/Info] Process of the frames : ",frame_file)
            print("----------------------------------------------------")
            process_FaceLandMark_from_container(input_path,output)
            # Measure execution time and print it
            end_time = time.time()
            execution_time = end_time - start_time
            print("[AU/Info] Execution time: ", round(execution_time*1000, 1), " ms")
            print("\n")


    # Delete the frames folder and its contents
    shutil.rmtree(tempfolder)

    # Measure total execution time
    end_time = time.time()
    execution_time_all_pross = end_time - start_time_all_pross

    print(f"[AU/Del] Deleted cache folder '{tempfolder}'")
    print("--------------------------------------------------------------")
    print("[AU/Info] Processing complete. All frames have been processed.")
    print("[AU/Info] Total frames processed: ", frame_count)
    print("[AU/Info] Total Execution time: ", round(execution_time_all_pross, 1), " seconds")
    print("--------------------------------------------------------------")


def extract_subclip(video_path, start_time, duration, output_path):
    cmd = [
        "ffmpeg",
        "-ss", str(start_time),
        "-i", video_path,
        "-t", str(duration),
        "-c:v", "libx264",
        "-c:a", "aac",
        "-y",
        output_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def process_AU_for_segments(csv_path, video_path, output_root):
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for idx, row in enumerate(reader):
            start = float(row['start_time'])
            segment_name = f"{os.path.splitext(os.path.basename(video_path))[0]}_segment_{idx}"
            
            os.makedirs(TEMP_CLIPS_DIR, exist_ok=True)
            temp_clip_path = os.path.join(TEMP_CLIPS_DIR, f"{segment_name}.mp4")
            output_folder = os.path.join(output_root, segment_name)
            os.makedirs(output_folder, exist_ok=True)

            print(f"[AU] Processing of segment {idx} - {start:.2f}s à {start+0.5:.2f}s")

            try:
                duration = float(row['end_time']) - start
                extract_subclip(video_path, start, duration, temp_clip_path)
                process_FaceLandMark_video(temp_clip_path, output_folder, seconds=0.5)
            except Exception as e:
                print(f"[!] Error on segment {idx}: {e}")
        print(f"[AU] Processing of segments completed")
    # Clean up temporary clips
    print(f"[AU/Del] Deleting temporary clips in '{TEMP_CLIPS_DIR}'")
    shutil.rmtree(TEMP_CLIPS_DIR)


def extract_openface_features(video_path, output_dir):
    openface_path = r"D:/Users/Gaspard/OpenFace/FaceLandmarkVidMulti.exe"
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        openface_path,
        "-f", video_path,
        "-out_dir", output_dir,
        '-aus',
        '-tracked'
    ]
    
    print(f"[OpenFace] Processing video: {video_path}")
    subprocess.run(cmd, check=True)
    print(f"[OpenFace] Output saved to: {output_dir}")

def detect_who_speaking_from_clips(video_id, segments_csv_path, openface_dir, output_path):
    segments = pd.read_csv(segments_csv_path)
    dyadic_segments = segments[segments["category"] == "dyadic"].reset_index(drop=True)
    dyadic_segments = dyadic_segments[dyadic_segments["end_time"] - dyadic_segments["start_time"] >= 5].reset_index(drop=True)

    results = []
    for i, row in dyadic_segments.iterrows():
        clip_name = f"{row['start_time']}_to_{row['end_time']}_segment.mp4"
        au_csv_path = os.path.join(openface_dir, f"{row['start_time']}_to_{row['end_time']}_segment.csv")
        speaker = row["speaker"]
        
        try:
            df = load_au_csv_with_defaults(au_csv_path)
            df.columns = df.columns.str.strip()
            face_activity = (
                df.groupby("face_id")[["AU25_r", "AU26_r", "AU27_r"]]
                .mean()
                .sum(axis=1)
            )
            if not face_activity.empty:
                face_id = face_activity.idxmax()
                results.append({"clip": clip_name, "speaker": speaker, "face_id": face_id})
                print(f"[✓] {clip_name}: {speaker} → face_id {face_id}")
            else:
                print(f"[!] No AU data for {clip_name}")
        except Exception as e:
            print(f"[✗] Error on {clip_name}: {e}")

    if results:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pd.DataFrame(results).to_csv(output_path, index=False)
        print(f"[✓] Mapping saved to {output_path}")

def run_openface_on_all_clips(clips_dir, openface_out_dir):
    os.makedirs(openface_out_dir, exist_ok=True)
    for clip in sorted(os.listdir(clips_dir)):
        if not clip.endswith(".mp4"):
            continue
        clip_path = os.path.join(clips_dir, clip)
        output_clip_dir = openface_out_dir
        csv_output = os.path.join(output_clip_dir, clip.replace(".mp4", ".csv"))
        if os.path.exists(csv_output):
            print(f"[OpenFace] Already processed: {clip}")
            continue
        print(f"[OpenFace] Processing {clip}")
        # subprocess.run([
        #     r"D:/Users/Gaspard/OpenFace/FaceLandmarkVidMulti.exe",
        #     "-f", clip_path,
        #     "-out_dir", openface_out_dir,
        #     '-aus',
        #     '-tracked'
        # ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        process_FaceLandmarkVidMulti_from_container(clip_path,openface_out_dir)
        


EXPECTED_AUS = ["AU25_r", "AU26_r", "AU27_r"]

def load_au_csv_with_defaults(csv_path):
    df = pd.read_csv(csv_path)

    for au in EXPECTED_AUS:
        if au not in df.columns:
            df[au] = 0.0

    return df