import cv2
import os
from actionUnitExtract import process_FaceLandMark_from_container
import shutil
import time

start_time_all_pross=time.time()

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
    print(f"[PreProcess] Frames saved in '{tempfolder}'")

    # Process all frames
    for frame_file in sorted(os.listdir(tempfolder)):
        if frame_file.endswith('.jpg'):
            start_time = time.time()
            input_path = os.path.join(tempfolder, frame_file)
            print("----------------------------------------------------")
            print("[Info] Process of the frames : ",frame_file)
            print("----------------------------------------------------")
            process_FaceLandMark_from_container(input_path,output)
            # Measure execution time and print it
            end_time = time.time()
            execution_time = end_time - start_time
            print("[Info] Execution time: ", round(execution_time*1000, 1), " ms")
            print("\n")


    # Delete the frames folder and its contents
    shutil.rmtree(tempfolder)

    # Measure total execution time
    end_time = time.time()
    execution_time_all_pross = end_time - start_time_all_pross

    print(f"[Del] Deleted folder '{tempfolder}'")
    print("--------------------------------------------------------------")
    print("[Info] Processing complete. All frames have been processed.")
    print("[Info] Total frames processed: ", frame_count)
    print("[Info] Total Execution time: ", round(execution_time_all_pross, 1), " seconds")
    print("--------------------------------------------------------------")



