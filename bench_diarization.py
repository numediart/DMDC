from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate
import pympi
import pandas as pd
from pathlib import Path
import re

def load_eaf_as_annotation(eaf_path):
    eaf = pympi.Elan.Eaf(str(eaf_path))
    annotation = Annotation()
    for tier in eaf.get_tier_names():
        if tier.startswith("SPEAKER_"):
            for start, end, _ in eaf.get_annotation_data_for_tier(tier):
                # Convert ms to s
                segment = Segment(start / 1000.0, end / 1000.0)
                annotation[segment] = tier
    return annotation

def load_csv_as_annotation(csv_path):
    df = pd.read_csv(csv_path)
    annotation = Annotation()
    for _, row in df.iterrows():
        segment = Segment(float(row['start_time']), float(row['end_time']))
        speaker = str(row['speaker'])
        annotation[segment] = speaker
    return annotation

def evaluate_diarization(eaf_path, csv_path):
    print(f"\n=== Évaluation DER pour : {eaf_path.name} vs {csv_path.name} ===")

    # Charger la vérité terrain depuis ELAN
    eaf_obj = pympi.Elan.Eaf(str(eaf_path))
    reference = Annotation()
    for tier in eaf_obj.get_tier_names():
        if 'SPEAKER_' in tier:
            for start, end, _ in eaf_obj.get_annotation_data_for_tier(tier):
                segment = Segment(start / 1000, end / 1000)  # ms → s
                reference[segment] = tier

    # Charger les prédictions depuis le CSV
    df = pd.read_csv(csv_path)
    hypothesis = Annotation()
    for _, row in df.iterrows():
        start = row['start_time']
        end = row['end_time']
        speaker = row['speaker']
        segment = Segment(start, end)
        hypothesis[segment] = speaker

    # Évaluer
    metric = DiarizationErrorRate()
    details = metric(reference, hypothesis, detailed=True)

    print(f"👉 DER: {details['diarization error rate']:.2%}")
    duration = reference.get_timeline().duration()
    print(f"Durée totale de parole dans la référence : {duration:.2f} secondes")

    for k in ['confusion', 'missed speech', 'false alarm']:
        if k in details:
            print(f"   - {k.capitalize()} : {100 * details[k] / duration:.2f}%")
        else:
            print(f"   - {k.capitalize()} : valeur non disponible")

# === Exemple d'utilisation ===

eaf_dir = Path("Benchmark/ELAN/Dataset_Bench_Manual")
csv_dir = Path("V0DataSet/segments")

for eaf_file in eaf_dir.glob("*.eaf"):
    base_name = eaf_file.stem.replace("_video", "")
    csv_file = csv_dir / f"{base_name}_segments.csv"
    if csv_file.exists():
        evaluate_diarization(eaf_file, csv_file)
    else:
        print(f"⚠️ Pas de CSV pour {eaf_file.name}")
