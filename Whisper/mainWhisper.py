import os
from transcriptFromAudio import transcriptFromAudio

output_path = os.path.join(os.path.dirname(__file__), "output")

for filename in os.listdir(os.path.join(os.path.dirname(__file__), "input")):
    if filename.endswith(".wav"):
        input_file = os.path.join(os.path.dirname(__file__), "input", filename)
        transcriptFromAudio(audiofile=input_file, outputFolder=output_path,modelType="tiny.en")