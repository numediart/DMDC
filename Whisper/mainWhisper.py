import os
from transcriptFromAudio import transcriptFromAudio

input_path = os.path.join(os.path.dirname(__file__), "input", "testclip_001.wav")
output_path = os.path.join(os.path.dirname(__file__), "output")

transcriptFromAudio(audiofile=input_path, outputFolder=output_path)