from pyannote.audio import Pipeline
import pandas as pd
import os
import torch
from pyannote.audio.pipelines.utils.hook import ProgressHook
from csvToSubTiltle import csv_to_subtitle
import time
#Constants




#Variables
filename= "3_video.wav"

def run_diarization(filename):
    start_time = time.time()
    ##################################################################
    #Credit https://github.com/pyannote/pyannote-audio README.md
    ##################################################################
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token="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")

    # send pipeline to GPU (when available)
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
        print("GPU available : ",torch.cuda.get_device_name(0))  
    else:
        print("GPU not available. Running on CPU.")



    # apply pretrained pipeline with 2 speakers
    with ProgressHook() as hook:
        diarization = pipeline("./V0DataSet/wav/"+filename,max_speakers=2,min_speakers=0,hook=hook)



    # Create a dataframe from the diarization results
    data = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        data.append({'start': turn.start, 'end': turn.end, 'speaker': speaker})
    df = pd.DataFrame(data)

    # Display the dataframe
    print(df)



    # Save dataframe as CSV with file name and timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(os.path.dirname("V0DataSet/Diarization_Results/"), exist_ok=True)
    pathname = f"./V0DataSet/Diarization_Results/{filename}_diarization_results_{timestamp}.csv"
    df.to_csv(pathname, index=False)
    print(f"Saved diarization results here: {pathname}")

    # Convert CSV as subtilefile and save it
    csv_to_subtitle(
        pathname,
        f"./V0DataSet/Subtitle/{filename}.srt"
    )




    # Measure execution time and print it
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time: ", execution_time, " seconds")
