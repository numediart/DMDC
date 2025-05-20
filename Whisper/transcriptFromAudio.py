import whisper
import pandas as pd
import os
import warnings
import time



#Warnings deletes
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

def transcriptFromAudio(audiofile,outputFolder,modelType="base.en"):
    """
        Transcribes an audio file into text and saves the transcription as a CSV file.

        Args:
            audiofile (str): Path to the audio file to be transcribed.
            outputFolder (str): Path to the folder where the output CSV file will be saved.
            modelType (str, optional): Type of Whisper model to use for transcription. 
                                       Defaults to "base.en".
    """
    #Start the timer for the monitoring
    start_time=time.time()


    audioFileName = os.path.splitext(os.path.basename(audiofile))[0]

    #Load the model
    model = whisper.load_model(modelType)
    #Run the model
    print("[Whisper] Transcription in progress...")
    result = model.transcribe(audiofile)

    #PRINTING/MONITORING Measure total execution time
    end_time = time.time()
    execution_time_all_pross = end_time - start_time
    print("[Whisper] Transcription done in ", round(execution_time_all_pross, 1), " seconds")


    #Take just the segments
    segments = result["segments"]
    data = [{"start": seg["start"], "end": seg["end"], "text": seg["text"]} for seg in segments]
    df = pd.DataFrame(data)
    output_file = os.path.join(outputFolder, audioFileName+"_transcript.csv")
    df.to_csv(output_file, index=False)
    print("[Whisper] Transcription save at: "+output_file)


