import subprocess
import os


command = ["docker", "ps"]
result = subprocess.run(command, capture_output=True, text=True)

# Extract the container ID
print("docker ps",result.stdout)
container_id = result.stdout.split("\n")[1][0:11]
print("Container ID:", container_id, "is running.")


def process_FaceLandMark_from_container(input_path):
    # Copy the image file into the container
    current_folder = os.path.abspath(os.path.dirname(__file__))
    print("input path", input_path)
    command = ["docker", "cp", input_path, f"{container_id}:/home/openface-build"]
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        print("Error copying file to container:", result.stderr)
        exit(1)

    print("File copied successfully.")
    # Extract the word between the last slash or backslash and .png
    file_name = os.path.basename(input_path)
    copiedFile = file_name.split(".png")[0]+".png"
    print("File copied is", copiedFile)

    # Run the FaceLandmarkImg command inside the container
    command = [
        "docker", "exec", "-it", container_id,
        "build/bin/FaceLandmarkImg", "-f", copiedFile
    ]
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        print("Error running FaceLandmarkImg:", result.stderr)
        exit(1)

    print("FaceLandmarkImg executed successfully.")


    # Copy the processed folder from the container to the local ./output directory
    output_parent_folder = os.path.join(current_folder, "output")
    os.makedirs(output_parent_folder, exist_ok=True)
    processed_folder_name = f"{copiedFile}_processed"
    output_folder = os.path.join(output_parent_folder, processed_folder_name)
    os.makedirs(output_folder, exist_ok=True)

    command = ["docker", "cp", f"{container_id}:/home/openface-build/processed", output_folder]
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        print("Error copying processed folder from container:", result.stderr)
        exit(1)

    print("Processed folder copied successfully to .",output_folder,"output_folder")

    # Delete the processed folder inside the container to clean up
    command = ["docker", "exec", "-it", container_id, "rm", "-rf", "/home/openface-build/processed"]
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        print("Error deleting processed folder inside the container:", result.stderr)
        exit(1)

    print("Processed folder deleted successfully inside the container.")
    return True

input_path = os.path.join(os.path.dirname(__file__), "input", "image1.png")
process_FaceLandMark_from_container(input_path)