import whisper
import pandas as pd
import os
import warnings
import time



#Warnings deletes
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

def transcriptFromAudio(audiofile, outputFolder, listTimeStamp="", modelType="base.en"):
    """
        Transcribes an audio file into text and saves the transcription as a CSV file.

        Args:
            audiofile (str): Path to the audio file to be transcribed.
            outputFolder (str): Path to the folder where the output CSV file will be saved.
            listTimeStamp(list(str),optional): List of the video timeStamp.
            modelType (str, optional): Type of Whisper model to use for transcription. 
                                       Defaults to "base.en".
        Available models:
        -----------------------------------------------------------------
        | Size   | Parameters | English-only model | Multilingual model | Required VRAM | Relative speed |
        -----------------------------------------------------------------
        | tiny   | 39 M       | tiny.en            | tiny               | ~1 GB         | ~10x           |
        | base   | 74 M       | base.en            | base               | ~1 GB         | ~7x            |
        | small  | 244 M      | small.en           | small              | ~2 GB         | ~4x            |
        | medium | 769 M      | medium.en          | medium             | ~5 GB         | ~2x            |
        | large  | 1550 M     | N/A                | large              | ~10 GB        | 1x             |
        | turbo  | 809 M      | N/A                | turbo              | ~6 GB         | ~8x            |
        -----------------------------------------------------------------
    """
    #Start the timer for the monitoring
    start_time=time.time()


    audioFileName = os.path.splitext(os.path.basename(audiofile))[0]

    #Load the model
    model = whisper.load_model(modelType)
    #Run the model
    print("[Whisper] Transcription in progress...")
    if listTimeStamp=="":
        result = model.transcribe(audiofile)
    else:
        # result = model.transcribe(audiofile, initial_prompt="",word_timestamps=True)
        result = model.transcribe(audiofile, clip_timestamps=listTimeStamp)


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
    return df


