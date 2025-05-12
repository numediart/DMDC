from pyannote.audio import Pipeline
import pandas as pd
import os
import time
import torch


start_time = time.time()



pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="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")

# # send pipeline to GPU (when available)
if torch.cuda.is_available():
    pipeline.to(torch.device("cuda"))
    print("GPU available : ",torch.cuda.get_device_name(0))  # should say GTX 1070
else:
    print("GPU not available. Running on CPU.")

# apply pretrained pipeline
filename= "2_video.wav"
diarization = pipeline("./V0DataSet/wav/"+filename,max_speakers=2,min_speakers=2)

# Create a dataframe from the diarization results
data = []
for turn, _, speaker in diarization.itertracks(yield_label=True):
    data.append({'start': turn.start, 'end': turn.end, 'speaker': speaker})
df = pd.DataFrame(data)

# Display the dataframe
print(df)

# Save dataframe as CSV with file name and timestamp
timestamp = time.strftime("%Y%m%d-%H%M%S")
pathname = f"./V0DataSet/Diarization_Results/{filename}_diarization_results_{timestamp}.csv"
df.to_csv(pathname, index=False)
print(f"Saved diarization results here: {pathname}")


# Measure execution time
end_time = time.time()
execution_time = end_time - start_time
print("Execution time: ", execution_time, " seconds")
