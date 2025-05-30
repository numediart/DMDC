import os
import cv2
import pandas as pd
from datetime import datetime


# Benchmark Parameter and path
inputBenchVideo = os.path.join(os.path.dirname(__file__), "../../V0DataSet/mp4/")
inputBenchSegments = os.path.join(os.path.dirname(__file__), "../../V0DataSet/segments/")
outputBenchImage = os.path.join(os.path.dirname(__file__), "./output/")


# We are not using it
def generate_segments(video, segments_file, max_size=128):
    """
    Generates video segments of a given maximum size and assigns a category to each segment based on overlap with provided segment annotations.
    Args:
        video (str): Name of the video file.
        segments_file (pd.DataFrame): DataFrame with 'start_time', 'end_time', and 'category' columns.
        max_size (int, optional): Maximum number of frames per segment. Defaults to 128.
    Returns:
        pd.DataFrame: DataFrame with 'start_time', 'end_time', and 'category' for each segment.
    """

    video_path = os.path.join(inputBenchVideo, video)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    segments = []
    num_segments = (total_frames + max_size - 1) // max_size
    # Assign type for each segment based on overlap with segments_file
    for seg_idx in range(num_segments):
        start_frame = seg_idx * max_size
        end_frame = min(start_frame + max_size - 1, total_frames - 1)
        start_time = start_frame / fps
        end_time = end_frame / fps

        # Find all categories that overlap with this segment
        overlapping = segments_file[
            (segments_file['end_time'] > start_time) & (segments_file['start_time'] < end_time)
        ]
        if not overlapping.empty:
            # If multiple, join categories with '+'
            video_type = '+'.join(sorted(set(overlapping['category'])))
        else:
            video_type = "unknown"

        segments.append({'start_time': start_time, 'end_time': end_time, 'category': video_type})
    # print(segments)
    return pd.DataFrame(segments)


def play_segment(cap, start_frame, end_frame, fps, play_speed=2):
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx > end_frame:
            break
        if frame_idx >= start_frame:
            # Resize to 480p (854x480)
            frame_480p = cv2.resize(frame, (854, 480))
            cv2.imshow('Video', frame_480p)
            # Play at play_speed: reduce waitKey delay (original: 1000/fps ms)
            delay = max(1, int((1000 / fps) / play_speed))
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break
        frame_idx += 1
    cap.release()
    cv2.destroyAllWindows()

def mainBenchImage(play_speed=2):
    """
    Runs a benchmarking process for video segments by displaying frames to the user for manual validation.
    Collects user input on the validity of each segment and saves the results to a CSV file.
    """

    benchmarkTable=[]
    for video in os.listdir(inputBenchVideo):
        id=video.split("_")[0]
        segments_file=pd.read_csv(os.path.join(inputBenchSegments,str(id)+"_segments.csv"))
        # Cut segments in smaller filer
        # segments_file = generate_segments(video, segments_file)
        print(segments_file)
        segments_file_size = len(segments_file)
        for _, segment in segments_file.iterrows():
            # print(video)
            start_sec = segment['start_time']
            end_sec = segment['end_time']
            video_type = segment['category']

            cap = cv2.VideoCapture(os.path.join(inputBenchVideo, video))
            fps = cap.get(cv2.CAP_PROP_FPS)
            start_frame = int(start_sec * fps)
            end_frame = int(end_sec * fps)

            print(
                "\n"*20,
                "="*60,
                "\n",
                "\t\t[Benchmark Category]",
                "\n",
                "\t ℹ️  Video N°", str(id),"Video Name :", video,"segments n°:",_,"/",str(segments_file_size),"\n ",
                "\t ⏳ Start sec=", round(start_sec,1),"(Frame n°", start_frame, ")\n",
                "\t ⌛ End sec=", round(end_sec,1),"(Frame n°", end_frame, ")\n",
                "\t 📼 Video_typ: ", video_type,
                "\n",
                "="*60
            )


            play_segment(cap, start_frame, end_frame, fps, play_speed)


            # USER INPUT "while"  0️⃣: false, 1️⃣: true, 2️⃣: replay, 3️⃣: save)
            while True:
                user_input_for_validation = input(
                    "The video was valid & only " + str(video_type) +
                    " or not (0️⃣: false, 1️⃣: true, 2️⃣: replay(slow motion), 3️⃣: save) : "
                )
                if user_input_for_validation == "1":
                    user_input_for_validation = True
                    break
                elif user_input_for_validation == "2":
                    # Replay the segment
                    cap_replay = cv2.VideoCapture(os.path.join(inputBenchVideo, video))
                    play_segment(cap_replay, start_frame, end_frame, fps, 0.7)
                    cap_replay.release()
                elif user_input_for_validation == "3":
                    # Save benchmark table so far
                    df_bench = pd.DataFrame(benchmarkTable)
                    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    df_bench.to_csv(os.path.join(outputBenchImage, "benchmark_results_" + now + ".csv"))
                    print("Benchmark table saved.")
                else:
                    user_input_for_validation = False
                    break

            print("You entered:", user_input_for_validation)

            benchmarkTable.append(
            {
                "video": video,
                "start_time": start_sec,
                "end_time": end_sec,
                "type": video_type,
                "valid": user_input_for_validation,
            }
            )
        cap.release()
        cv2.destroyAllWindows()

        user_input_for_validation = input("The videos was valid & only "+video_type+" or not (0: false, 1: true) : ")
        if user_input_for_validation=="1":
            user_input_for_validation=True
        else:
            user_input_for_validation=False
    
        print("You entered:", user_input_for_validation)
        benchmarkTable.append(
            { 
                "video":video,
                "start_time":start_sec,
                "end_time":end_sec,
                "type":video_type,
                "valid":user_input_for_validation,
        }
        )


    # Benchmark export
    df_bench=pd.DataFrame(benchmarkTable)
    print(df_bench)
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    df_bench.to_csv(os.path.join(outputBenchImage, "benchmark_results_" + now + ".csv"))


mainBenchImage()