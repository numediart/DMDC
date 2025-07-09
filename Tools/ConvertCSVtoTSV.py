import pandas as pd

for i in range(13, 15):
    csv_file = f"V0DataSet/segments/{i}_segments.csv"
    df = pd.read_csv(csv_file, quotechar='"')

    tsv_file = f"{i}_segments.tsv"
    df.to_csv(tsv_file, sep="\t", index=False)

for i in range(13, 15):
    csv_file = f"V0DataSet/Diarization_Results/{i}_video_diarization_results.csv"   
    df = pd.read_csv(csv_file, quotechar='"')

    tsv_file = f"{i}_diarization.tsv"
    df.to_csv(tsv_file, sep="\t", index=False)