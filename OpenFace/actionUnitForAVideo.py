import cv2
import os
from actionUnitExtract import process_FaceLandMark_from_container
import shutil

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
    print(f"Frames saved in '{tempfolder}'")

    # Process all frames
    for frame_file in sorted(os.listdir(tempfolder)):
        if frame_file.endswith('.jpg'):
            input_path = os.path.join(tempfolder, frame_file)
            process_FaceLandMark_from_container(input_path,output)

    # Delete the frames folder and its contents
    shutil.rmtree(tempfolder)
    print(f"Deleted folder '{tempfolder}'")


