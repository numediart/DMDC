import cv2
import os
from actionUnitExtract import process_FaceLandMark_from_container
import shutil

# Path to the video file
video_path = './input/2_video.mp4'
output_folder = 'frames'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Get the frame rate of the video
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = 0

# Process the first second of the video
seconds=10
while frame_count < fps*seconds:
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