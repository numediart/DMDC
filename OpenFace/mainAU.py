from actionUnitExtract import process_FaceLandMark_from_container
from actionUnitForAVideo import process_FaceLandMark_video
import os

#example usage for process_FaceLandMark_from_container
input_path = os.path.join(os.path.dirname(__file__), "input", "test_img1.png")
output_path = os.path.join(os.path.dirname(__file__), "output")

# process_FaceLandMark_from_container(input_path,output_path,onlyCSVOutput=False)

# Example usage for process_FaceLandMark_video
video_path = "./input/test_video1.mp4"  # Replace with the path to your video file
output = "./output/test_video1/"    # Temporary folder to store frames

# process_FaceLandMark_video(video_path,output,seconds=0.5 )