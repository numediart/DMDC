import subprocess
import os

#Check if at least one container is running
command = ["docker", "ps"]
result = subprocess.run(command, capture_output=True, text=True)

lines = result.stdout.strip().split("\n")

# Check that there is at least 2 container
if len(lines) < 2:
    raise RuntimeError("Aucun conteneur actif trouvé.")

first_container_line = lines[1]

# Remove space
container_id = first_container_line.split()[0]

print("[Docker/Info] Container ID:", container_id, "is running.")


def process_FaceLandMark_from_container(input_path, output_path,onlyCSVOutput=True):
    """
    Processes a facial landmark image using a Docker container.
    Args:
        input_path (str): The path to the input image file.
        output_path (str): The path to the output directory.
    Returns:
        bool: True if the process completes successfully.
    Steps:
        1. Copies the input image file into the Docker container.
        2. Executes the FaceLandmarkImg command inside the container.
        3. Copies the processed output folder from the container to the specified output directory.
        4. Cleans up by deleting the processed folder inside the container.
    """

    # Copy the image file into the container
    command = ["docker", "cp", input_path, f"{container_id}:/home/openface-build"]
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        print("[ERR] Error copying file to container:", result.stderr)
        exit(1)

    # Extract the word between the last slash or backslash and .png
    file_name = os.path.basename(input_path)
    copiedFile = file_name.split("/")[-1]
    print("[OpenFace/Copy] File(",copiedFile,") copied successfully.")

    # Run the FaceLandmarkImg command inside the container
    command = [
        "docker", "exec", "-it", container_id,
        "build/bin/FaceLandmarkImg", "-f", copiedFile
    ]
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        print("Error running FaceLandmarkImg:", result.stderr)
        exit(1)

    print("[OpenFace/Info] FaceLandmarkImg executed successfully.")

    # Copy the processed folder from the container to the specified output directory
    os.makedirs(output_path, exist_ok=True)
    processed_folder_name = f"{copiedFile}_processed"


    if(onlyCSVOutput==True):#Just the CSV
        output_folder = os.path.join(output_path)
        os.makedirs(output_folder, exist_ok=True)   
        csv_file_name = os.path.splitext(copiedFile)[0] + ".csv"
        command = ["docker", "cp", f"{container_id}:/home/openface-build/processed/{csv_file_name}", output_folder]
    elif(onlyCSVOutput==False):#All the file
        output_folder = os.path.join(output_path, processed_folder_name)
        os.makedirs(output_folder, exist_ok=True)
        command = ["docker", "cp", f"{container_id}:/home/openface-build/processed", output_folder]
        
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        print("[ERR] Error copying processed folder from container:", result.stderr)
        exit(1)

    print("[CP] Processed folder copied successfully to", output_folder)

    # Delete the processed folder inside the container to clean up
    command = ["docker", "exec", "-it", container_id, "rm", "-rf", "/home/openface-build/processed"]
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        print("[ERR] Error deleting processed folder inside the container:", result.stderr)
        exit(1)

    print("[OpenFace/Del] Processed folder deleted successfully inside the container.")
    return True

