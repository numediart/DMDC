from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate
import pympi
import pandas as pd
from pathlib import Path

def load_eaf_as_annotation(eaf_path):
    eaf = pympi.Elan.Eaf(str(eaf_path))
    annotation = Annotation()
    for tier in eaf.get_tier_names():
        if tier.startswith("SPEAKER_"):
            for start, end, _ in eaf.get_annotation_data_for_tier(tier):
                segment = Segment(start / 1000.0, end / 1000.0)  # Convert ms to seconds
                annotation[segment] = tier
    return annotation

def load_csv_as_annotation(csv_path):
    df = pd.read_csv(csv_path)
    annotation = Annotation()
    # Handle both start/end and start_time/end_time columns
    if 'start' in df.columns and 'end' in df.columns:
        start_col, end_col = 'start', 'end'
    elif 'start_time' in df.columns and 'end_time' in df.columns:
        start_col, end_col = 'start_time', 'end_time'
    else:
        raise ValueError(f"CSV columns not recognized: {df.columns}")
    for _, row in df.iterrows():
        segment = Segment(float(row[start_col]), float(row[end_col]))
        speaker = str(row['speaker'])
        annotation[segment] = speaker
    return annotation

def evaluate_diarization(eaf_path, csv_path, results):
    print(f"\n=== DER Evaluation: {eaf_path.name} vs {csv_path.name} ===")

    # Load reference annotations from ELAN
    reference = load_eaf_as_annotation(eaf_path)

    # Load hypothesis annotations from CSV
    hypothesis = load_csv_as_annotation(csv_path)

    # Evaluate DER
    metric = DiarizationErrorRate()
    details = metric(reference, hypothesis, detailed=True)

    der = details['diarization error rate']
    duration = reference.get_timeline().duration()

    print(f"👉 DER: {der:.2%}")
    print(f"Reference speech duration: {duration:.2f} seconds")

    confusion = details.get("confusion", None)
    missed = details.get("missed detection", None)
    false_alarm = details.get("false alarm", None)

    if confusion is not None:
        confusion_pct = 100 * confusion / duration
        print(f"   - Confusion: {confusion_pct:.2f}%")
    else:
        confusion_pct = None
        print("   - Confusion: not available")

    if missed is not None:
        missed_pct = 100 * missed / duration
        print(f"   - Missed speech: {missed_pct:.2f}%")
    else:
        missed_pct = None
        print("   - Missed speech: not available")

    if false_alarm is not None:
        false_alarm_pct = 100 * false_alarm / duration
        print(f"   - False alarm: {false_alarm_pct:.2f}%")
    else:
        false_alarm_pct = None
        print("   - False alarm: not available")

    # Append results to the list
    results.append({
        "file": eaf_path.name,
        "DER": der,
        "Confusion (%)": confusion_pct,
        "Missed Speech (%)": missed_pct,
        "False Alarm (%)": false_alarm_pct,
        "Reference Duration (s)": duration
    })


# === Evaluation loop and CSV export ===

eaf_dir = Path("Benchmark/ELAN/Dataset_Bench_Manual")

# csv_dir = Path("V0DataSet/segments")
# csv_dir = Path("TitaNet-LargeDiarization/output")
# csv_dir = Path("PyannoteDiarizationV2.1/output")
csv_dir = Path("V0.12DataSet/Diarization_Results")
results = []

for eaf_file in eaf_dir.glob("*.eaf"):
    base_name = eaf_file.stem.replace("_video", "")
    csv_file = csv_dir / f"{base_name}_video_diarization_results.csv"
    # csv_file = csv_dir / f"{base_name}_video_diarization.csv"
    # csv_file = csv_dir / f"{base_name}_segments.csv"
    print(csv_file)

    if csv_file.exists():
        evaluate_diarization(eaf_file, csv_file, results)
    else:
        print(f"⚠️ No CSV found for {eaf_file.name}")

# Save evaluation results to CSV
results_df = pd.DataFrame(results)
results_df.sort_values(by="file", inplace=True)
results_df.to_csv("diarization_results.csv", index=False)
print("\n✅ All evaluations completed. Results saved to diarization_results.csv.")
