import pandas as pd
import os

def format_all_clips(mapping_csv, openface_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    mapping_df = pd.read_csv(mapping_csv)

    for _, row in mapping_df.iterrows():
        clip_name = row['clip']
        speaker_face_id = row['face_id']
        
        clip_base = os.path.splitext(clip_name)[0]
        csv_path = os.path.join(openface_dir, f"{clip_base}.csv")

        speaker_out_path = os.path.join(output_dir,"speaker", f"{clip_base}_speaker.csv")
        listener_out_path = os.path.join(output_dir, 'listener',f"{clip_base}_listener.csv")
        os.makedirs(os.path.join(output_dir,"speaker"), exist_ok=True)
        os.makedirs(os.path.join(output_dir,"listener"), exist_ok=True)
        
        if not os.path.exists(csv_path):
            print(f"[✗] Missing OpenFace CSV: {csv_path}")
            continue

        try:
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()
            df = df[df['success'] == 1]

            drop_cols = ['face_id', 'timestamp', 'confidence', 'success']

            # Keep only frames where both face_id 0 and 1 are present
            frame_counts = df['frame'].value_counts()
            valid_frames = frame_counts[frame_counts >= 2].index
            df = df[df['frame'].isin(valid_frames)]
            
            speaker_df = df[df['face_id'] == speaker_face_id].copy()
            listener_df = df[df['face_id'] != speaker_face_id].copy()

            if speaker_df.empty or listener_df.empty:
                print(f"[✗] Skipped {clip_name}: Missing speaker or listener")
                continue

            speaker_df = speaker_df.drop(columns=drop_cols, errors='ignore')
            listener_df = listener_df.drop(columns=drop_cols, errors='ignore')

            speaker_df.to_csv(speaker_out_path, index=False)
            listener_df.to_csv(listener_out_path, index=False)

            print(f"[✓] Saved: {clip_base}_speaker.csv and {clip_base}_listener.csv")

        except Exception as e:
            print(f"[✗] Error with {clip_name}: {e}")

