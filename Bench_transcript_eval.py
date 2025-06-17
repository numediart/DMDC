import os
import pandas as pd
from jiwer import compute_measures, cer, wer

# Folder containing the files to evaluate
folder_path = "test/clean_segments/"
csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

results = []

print("===== EVALUATION BY FILE =====\n")

for filename in csv_files:
    file_path = os.path.join(folder_path, filename)

    try:
        df = pd.read_csv(file_path)
        df = df.dropna(subset=["text", "transcription"])

        if df.empty:
            print(f"{filename}: empty file after cleaning.")
            continue

        refs = df["text"].astype(str).str.strip().tolist()
        hyps = df["transcription"].astype(str).str.strip().tolist()

        joined_refs = " ".join(refs)
        joined_hyps = " ".join(hyps)

        wer_val = wer(joined_refs, joined_hyps)
        cer_val = cer(joined_refs, joined_hyps)
        measures = compute_measures(joined_refs, joined_hyps)
        total_words = len(joined_refs.split())

        print(f"--- {filename} ---")
        print(f"WER: {wer_val:.2%}")
        print(f"CER: {cer_val:.2%}")
        
        print(f"Total words: {total_words}")
        print(f"S: {measures.get('substitutions', 'N/A')} | I: {measures.get('insertions', 'N/A')} | D: {measures.get('deletions', 'N/A')}\n")

        # Secure the addition to the result table
        try:
            results.append({
                "file": filename,
                "wer": wer_val,
                "cer": cer_val,
                "substitutions": measures.get("substitutions", -1),
                "insertions": measures.get("insertions", -1),
                "deletions": measures.get("deletions", -1),
                "total_words": total_words
            })
        except Exception as e_append:
            print(f"Error while adding {filename} to results: {e_append}")

    except Exception as e:
        print(f"Error processing {filename}: {e}")

# Check if any results were collected
if results:
    results_df = pd.DataFrame(results)
    results_df.to_csv("transcription_results.csv", index=False)
    print("\n✅ Results saved to transcription_results.csv")
else:
    print("\n⚠️ No results collected. Check the input files.")
