import pympi
import pandas as pd
from sklearn.metrics import classification_report
import jiwer
from jiwer import Compose, ToLowerCase, RemovePunctuation, RemoveMultipleSpaces, Strip, compute_measures
import os
from pathlib import Path
import re


def run_benchmark(eaf_path, csv_path):
    print(f"Benchmarking {eaf_path} with {csv_path}")
    eaf_obj = pympi.Elan.Eaf(eaf_path)

    # Mapping only for SkinDeep videos
    # video_number_match = re.match(r"(\d+)_video", eaf_path.stem)
    # apply_mapping = False
    # if video_number_match:
    #     video_number = int(video_number_match.group(1))
    #     apply_mapping = 6 <= video_number <= 10

    # if apply_mapping:
    #     speaker_mapping = {
    #         "SPEAKER_00": "SPEAKER_02",
    #         "SPEAKER_01": "SPEAKER_00",
    #         "SPEAKER_02": "SPEAKER_01",
    #     }
    # else:
    #     speaker_mapping = {}

    annotations_manuelles = []
    tier_names = eaf_obj.get_tier_names()

    for tier_name in tier_names:
        annotations_data = eaf_obj.get_annotation_data_for_tier(tier_name)
        
        # if 'SPEAKER_' in tier_name:
        #     for start, end, transcript_text in annotations_data:
        #         annotations_manuelles.append({
        #             'type': 'transcription',
        #             'speaker_id': tier_name,  # ex: 'SPEAKER_00'
        #             'start_ms': start,
        #             'end_ms': end,
        #             'label': transcript_text
        #         })
                
        # el
        if tier_name == 'category':
            for start, end, category_type in annotations_data:
                annotations_manuelles.append({
                    'type': 'category',
                    'speaker_id': None,
                    'start_ms': start,
                    'end_ms': end,
                    'label': category_type
                })

    # print(f"Trouvé {len(annotations_manuelles)} annotations manuelles au total.")
    # print(f"Annotations manuelles: {annotations_manuelles[:5]}")

    # print("\n---\n")
    # print("---\n")
    # print("---\n\n")

    df_auto = pd.read_csv(csv_path)

    annotations_auto = []
    for index, row in df_auto.iterrows():
        # Seconds to milliseconds to match ELAN format
        start_ms = row['start_time'] * 1000
        end_ms = row['end_time'] * 1000
        

        # annotations_auto.append({
        #     'type': 'transcription',
        #     'speaker_id': row['speaker'],
        #     'start_ms': start_ms,
        #     'end_ms': end_ms,
        #     'label': row['transcription']
        # })
        

        annotations_auto.append({
            'type': 'category',
            'start_ms': start_ms,
            'end_ms': end_ms,
            'label': row['category']
        })

    # print(f"Chargé et traité {len(df_auto)} lignes du CSV.")
    # print(f"Annotations automatiques: {annotations_auto[:5]}")

    # print("\n---\n")
    # print("---\n")
    # print("---\n\n")

    manual_categories = [ann for ann in annotations_manuelles if ann['type'] == 'category']

    auto_predicted_categories = [ann for ann in annotations_auto if ann['type'] == 'category']

    y_true = []
    y_pred = []

    for auto_cat_annot in auto_predicted_categories:
        found_manual_category = "None" # default value if no match found
        
        # Find the manual category that fully contains the automatic category segment
        for manual_cat_annot in manual_categories:
            # Check if the automatic category segment is fully contained within the manual category segment
            if (auto_cat_annot['start_ms'] >= manual_cat_annot['start_ms'] and
                auto_cat_annot['end_ms'] <= manual_cat_annot['end_ms']):
                found_manual_category = manual_cat_annot['label']
                break
        
        predicted_label = auto_cat_annot['label']
        
        y_true.append(found_manual_category)
        y_pred.append(predicted_label)

    print("--- Category classification ---")
    print(classification_report(y_true, y_pred, zero_division=0))



    unique_speakers = df_auto['speaker'].dropna().unique()
    all_speaker_ids = sorted([str(spk) for spk in unique_speakers])

    wer_results = {}
    # print("\n--- Word Error for each speaker ---")

    # for speaker in all_speaker_ids:
    #     # 2. Ground Truth ELAN
    #     manual_texts = sorted(
    #         [ann for ann in annotations_manuelles 
    #         if ann.get('speaker_id') == speaker_mapping.get(speaker, speaker)], 
    #         key=lambda x: x['start_ms']
    #     )
    #     ground_truth = " ".join([ann['label'] for ann in manual_texts])

    #     # 3. CSV Hypothesis
    #     df_speaker_auto = df_auto[df_auto['speaker'] == speaker].sort_values(by='start_time')
    #     hypothesis = " ".join(df_speaker_auto['transcription'].astype(str))
        
    #     # print(f"\n--- DÉBOGAGE POUR LE SPEAKER : {speaker} ---")
    #     # print(f"Nombre de mots (Vérité Terrain): {len(ground_truth.split())}")
    #     # print(f"Extrait Vérité Terrain: {ground_truth[:200]}")
    #     # print(f"Nombre de mots (Hypothèse)    : {len(hypothesis.split())}")
    #     # print(f"Extrait Hypothèse    : {hypothesis[:200]}")

    #     # 4. WER (Word Error Rate) Calculation
    #     if ground_truth:
    #         measures = jiwer.compute_measures(ground_truth, hypothesis)
    #         wer_results[speaker] = measures['wer']
    #         print(f"Speaker '{speaker}' - WER: {measures['wer']:.2%}")
    #     else:
    #         # If Speaker in CSV but not in EAF
    #         wer_results[speaker] = None
    #         print(f"Speaker '{speaker}' - No ground truth available, skipping WER calculation.")

    # # Mean WER
    # valid_wers = [w for w in wer_results.values() if w is not None]
    # if valid_wers:
    #     average_wer = sum(valid_wers) / len(valid_wers)
    #     print(f"\nMean WER (Macro-average): {average_wer:.2%}")
    
    return


# PATH
eaf_dir = Path("Benchmark/ELAN/Dataset_Bench_Manual")
csv_dir = Path("V0.7DataSet/segments_video_type/7_video")
csv_dir = Path("V0DataSet/segments/")

for eaf_file in eaf_dir.glob("*.eaf"):
    base_name = eaf_file.stem.replace("_video", "")
    csv_file = csv_dir / f"{base_name}_segments.csv"

    if csv_file.exists():
        run_benchmark(eaf_file, csv_file)
    else:
        print(f"⚠️ No CSV for {eaf_file.name}")