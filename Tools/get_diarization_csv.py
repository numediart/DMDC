import os
import glob

def get_diarization_csv(wav_path,dataset):
    base_filename = os.path.splitext(os.path.basename(wav_path))[0]  # e.g. "1_video"
    diarization_folder = os.path.join(dataset, "Diarization_Results")

    # Search for the CSV file that starts with the correct name
    pattern = os.path.join(diarization_folder, f"{base_filename}_diarization_results.csv")
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
    else:
        raise FileNotFoundError(f"No CSV file found for {base_filename} in {diarization_folder}")
    