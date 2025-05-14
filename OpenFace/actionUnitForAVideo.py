import cv2
import os
from actionUnitExtract import process_FaceLandMark_from_container
import shutil

def process_video(video_path, output_folder, seconds=1):
    """
    Extracts frames from a video.
    Args:
        video_path (str): Path to the input video file.
        output_folder (str): Directory to save extracted frames temporarily.
        seconds (int, optional): Number of seconds to process. Defaults to 1.
    """

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

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
        frame_filename = os.path.join(output_folder, f'frame_{frame_count:04d}.jpg')
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()
    print(f"Frames saved in '{output_folder}'")

    # Process all frames
    for frame_file in sorted(os.listdir(output_folder)):
        if frame_file.endswith('.jpg'):
            input_path = os.path.join(output_folder, frame_file)
            process_FaceLandMark_from_container(input_path)

    # Delete the frames folder and its contents
    shutil.rmtree(output_folder)
    print(f"Deleted folder '{output_folder}'")


# Example usage
video_path = "./input/2_video.mp4"  # Replace with the path to your video file
output_folder = "temp_frames"    # Temporary folder to store frames
seconds_to_process = 2         # Number of seconds to process

process_video(video_path, output_folder, seconds_to_process)