import subprocess
import os




def process_FaceLandmarkVidMulti_from_container(input_path, output_path,onlyCSVOutput=True,docker_usage=True):
   
    if(docker_usage):
        # Check if at least one container is running
        command = ["docker", "ps"]
        result = subprocess.run(command, capture_output=True, text=True)

        lines = result.stdout.strip().split("\n")

        # Check that there is at least 2 container
        if len(lines) < 2:
            print("[Docker/Error] No active container found.")
            print("[Docker/Hint] Please start a container using the following command:")
            print("               docker run -it --rm algebr/openface:latest")
            raise RuntimeError("No active container found.")

        first_container_line = lines[1]

        # Remove space
        container_id = first_container_line.split()[0]

        print("[Docker/Info] Container ID:", container_id, "is running.")


        # Copy the video file into the container
        command = ["docker", "cp", input_path, f"{container_id}:/home/openface-build"]
        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0:
            print("[ERR] Error copying file to container:", result.stderr)
            exit(1)

        # Extract the word between the last slash or backslash and .png
        file_name = os.path.basename(input_path)
        copiedFile = file_name.split("/")[-1]
        print("[OpenFace/Copy] File(",copiedFile,") copied successfully.")

        # Run the FaceLandmarkVidMulti command inside the container
        command = [
            "docker", "exec", "-it", container_id,
            "build/bin/FaceLandmarkVidMulti", "-f", copiedFile, "-aus"
        ]
        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0:
            print("Error running FaceLandmarkVidMulti:", result.stderr)
            exit(1)

        print("[OpenFace/Info] FaceLandmarkVidMulti executed successfully.")

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


if __name__ == "__main__":
    # Example usage of the process_FaceLandMark_video_from_container function
    input_path = os.path.join(os.path.dirname(__file__), "input", "clip3.mp4")
    # input_path = os.path.join(os.path.dirname(__file__), "input", "1_video.mp4")
    output_path = os.path.join(os.path.dirname(__file__), "output")

    try:
        success = process_FaceLandmarkVidMulti_from_container(input_path, output_path, onlyCSVOutput=False)
        if success:
            print("[Main] Video processing completed successfully.")
    except RuntimeError as e:
        print("[Main/ERR]", str(e))
    except Exception as e:
        print("[Main/ERR] An unexpected error occurred:", str(e))
