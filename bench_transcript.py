import pympi
import pandas as pd
from pathlib import Path
import os
from pydub import AudioSegment
import subprocess
import sys

# Répertoires
video_dir = Path("V0DataSet/mp4")
csv_dir = Path("test/clean_segments")
tmp_wav_base = Path("V0DataSet/tmp_wav")

# Dossier contenant les fichiers EAF
eaf_dir = Path("Benchmark/ELAN/Dataset_Bench_Manual")
# Dossier de sortie pour les segments "clean"
output_dir = Path("test/clean_segments")
output_dir.mkdir(parents=True, exist_ok=True)

def extract_clean_segments(annotations):
    transcriptions = [ann for ann in annotations if ann['type'] == 'transcription']
    clean_segments = []

    for i, seg in enumerate(transcriptions):
        overlapping = False
        for j, other in enumerate(transcriptions):
            if i == j:
                continue
            if seg['speaker_id'] != other['speaker_id']:
                if not (seg['end_ms'] <= other['start_ms'] or seg['start_ms'] >= other['end_ms']):
                    overlapping = True
                    break
        if not overlapping:
            clean_segments.append(seg)

    return clean_segments

for eaf_file in eaf_dir.glob("*.eaf"):
    eaf_obj = pympi.Elan.Eaf(eaf_file)

    annotations = []
    for tier_name in eaf_obj.get_tier_names():
        if 'SPEAKER_' in tier_name:
            for start, end, text in eaf_obj.get_annotation_data_for_tier(tier_name):
                annotations.append({
                    'type': 'transcription',
                    'speaker_id': tier_name,
                    'start_ms': start,
                    'end_ms': end,
                    'label': text
                })

    clean_segments = extract_clean_segments(annotations)

    df = pd.DataFrame([
        {
            'speaker': seg['speaker_id'],
            'start_ms': seg['start_ms'],
            'end_ms': seg['end_ms'],
            'text': seg['label']
        }
        for seg in clean_segments
    ])

    base_name = eaf_file.stem.replace("_video", "")
    output_csv = output_dir / f"{base_name}_clean_segments.csv"
    df.to_csv(output_csv, index=False)

print("✅ Extraction terminée. CSV placés dans test/clean_segments/")

# Crée le dossier temporaire si nécessaire
tmp_wav_base.mkdir(parents=True, exist_ok=True)

# Fonction pour découper et enregistrer les segments
def create_audio_segments(csv_file):
    base_name = csv_file.stem.replace("_clean_segments", "")
    wav_output_dir = tmp_wav_base / base_name
    wav_output_dir.mkdir(parents=True, exist_ok=True)

    video_path = video_dir / f"{base_name}_video.mp4"
    if not video_path.exists():
        print(f"[WARN] Vidéo non trouvée : {video_path}")
        return None

    # Extraction audio avec ffmpeg en .wav mono 16kHz
    full_wav_path = wav_output_dir / f"{base_name}.wav"
    if not full_wav_path.exists():
        os.system(
            f"ffmpeg -i \"{video_path}\" -ar 16000 -ac 1 -y \"{full_wav_path}\" > /dev/null 2>&1"
        )

    # Charger l'audio extrait
    audio = AudioSegment.from_wav(full_wav_path)

    # Lire le CSV de segments
    df = pd.read_csv(csv_file)
    segment_paths = []

    for idx, row in df.iterrows():
        start_ms = int(row['start_ms'])
        end_ms = int(row['end_ms'])
        seg_audio = audio[start_ms:end_ms]
        seg_path = wav_output_dir / f"{idx:04d}.wav"
        seg_audio.export(seg_path, format="wav")
        segment_paths.append(str(seg_path))

    return base_name, segment_paths

# Boucle sur tous les CSV
for csv_file in csv_dir.glob("*_clean_segments.csv"):
    result = create_audio_segments(csv_file)
    if result is None:
        continue
    base_name, segment_paths = result

    # Appel subprocess vers transcribe_parakeet.py
    print(f"[INFO] Lancement de la transcription pour {base_name} avec {len(segment_paths)} segments.")
    subprocess.run([sys.executable, "transcribe_parakeet.py", base_name, *segment_paths])