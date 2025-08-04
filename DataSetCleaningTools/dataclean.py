import numpy as np
import os
import re
import csv

def create_DMDC_database_from_pipeline(dataset, output_dir, csv_file, video_index_start=1, video_index_end=10):
    all_listener_data = []
    all_speaker_data = []
    all_mfcc_data = []
    all_transcript_data = []
    csv_rows = []  # Collect rows for CSV output

    for index in range(video_index_start, video_index_end + 1):
        listener_dir = os.path.join('.', dataset, 'n_frames_windowed_clips', f'{index}_video', 'listener')
        speaker_dir = os.path.join('.', dataset, 'n_frames_windowed_clips', f'{index}_video', 'speaker')
        mfcc_dir = os.path.join('.', dataset, 'mfcc_output', f'{index}_video')
        transcript_dir = os.path.join('.', dataset, 'transcripts', f'{index}_video')

        def extract_base(filename):
            m = re.search(r'(\d+)_to_(\d+)', filename)
            return m.groups() if m else None

        def build_base_to_filename(folder):
            mapping = {}
            if not os.path.exists(folder):
                print(f"Warning: Directory {folder} does not exist.")
                return mapping
            for filename in sorted(os.listdir(folder)):  # Sort filenames alphabetically
                path = os.path.join(folder, filename)
                if not os.path.isfile(path):
                    continue
                base = extract_base(filename)
                if base:
                    mapping[base] = filename
            return mapping

        listener_map = build_base_to_filename(listener_dir)
        speaker_map = build_base_to_filename(speaker_dir)
        mfcc_map = build_base_to_filename(mfcc_dir)
        transcript_map = build_base_to_filename(transcript_dir)

        listener_set = set(listener_map.keys())
        speaker_set = set(speaker_map.keys())
        mfcc_set = set(mfcc_map.keys())
        transcript_set = set(transcript_map.keys())

        common = listener_set & speaker_set & mfcc_set & transcript_set

        print(f"Found {len(common)} complete quadruples.")

        for base in common:
            def read_file_content(filepath, is_npy=False):
                try:
                    if is_npy:
                        return np.load(filepath)
                    else:
                        with open(filepath, 'r') as file:
                            return file.read()
                except Exception as e:
                    print(f"Error reading file {filepath}: {e}")
                    return None

            listener_file = os.path.join(listener_dir, listener_map[base])
            speaker_file = os.path.join(speaker_dir, speaker_map[base])
            mfcc_file = os.path.join(mfcc_dir, mfcc_map[base])
            transcript_file = os.path.join(transcript_dir, transcript_map[base])

            listener_data = read_file_content(listener_file, is_npy=True)
            speaker_data = read_file_content(speaker_file, is_npy=True)
            mfcc_data = read_file_content(mfcc_file, is_npy=True)
            transcript_data = read_file_content(transcript_file, is_npy=False)

            if all(data is not None for data in (listener_data, speaker_data, mfcc_data, transcript_data)):
                all_listener_data.append(listener_data)
                all_speaker_data.append(speaker_data)
                all_mfcc_data.append(mfcc_data)
                all_transcript_data.append(transcript_data)
                start_frame, end_frame = base
                csv_rows.append([index, start_frame, end_frame, transcript_data, listener_file, speaker_file, mfcc_file, transcript_file])

    # Save numpy arrays
    if all_listener_data:
        np.save(os.path.join(output_dir, f"{dataset}_listener.npy"), np.array(all_listener_data, dtype=object))
        print(f"Saved listener numpy array to {os.path.join(output_dir, f'{dataset}_AU_listener.npy')}")

    if all_speaker_data:
        np.save(os.path.join(output_dir, f"{dataset}_speaker.npy"), np.array(all_speaker_data, dtype=object))
        print(f"Saved speaker numpy array to {os.path.join(output_dir, f'{dataset}_AU_speaker.npy')}")

    if all_mfcc_data:
        np.save(os.path.join(output_dir, f"{dataset}_mfcc.npy"), np.array(all_mfcc_data, dtype=object))
        print(f"Saved mfcc numpy array to {os.path.join(output_dir, f'{dataset}_mfcc.npy')}")

    if all_transcript_data:
        np.save(os.path.join(output_dir, f"{dataset}_transcript.npy"), np.array(all_transcript_data, dtype=object))
        print(f"Saved transcript numpy array to {os.path.join(output_dir, f'{dataset}_transcript.npy')}")

    # Save CSV file
    if csv_rows:
        csv_path = os.path.join(output_dir, csv_file)
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Video Index', 'Start Frame', 'End Frame', 'Transcript Data', 'listener_file', "speaker_file", "mfcc_file", "transcript_file"])
            writer.writerows(csv_rows)
        print(f"Saved CSV file to {csv_path}")
    else:
        print("No data found to save to CSV.")

if __name__ == "__main__":
    DATASET = "V0.10DataSet"         # Name of the dataset directory
    OUTPUT_DIR = "./output"          # Directory to save the output files
    CSV_FILE = "output.csv"          # Name of the output CSV file
    VIDEO_INDEX_START = 1            # Start index for video processing
    VIDEO_INDEX_END = 10             # End index for video processing

    print("Configuration:")
    print(f"  Dataset: {DATASET}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  CSV file: {CSV_FILE}")
    print(f"  Video index start: {VIDEO_INDEX_START}")
    print(f"  Video index end: {VIDEO_INDEX_END}")

    create_DMDC_database_from_pipeline(
        DATASET,
        OUTPUT_DIR,
        CSV_FILE,
        video_index_start=VIDEO_INDEX_START,
        video_index_end=VIDEO_INDEX_END
    )
