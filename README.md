# DMDC README
*WIP README*

## Project Description

The goal of this project is to reproduce the data creation process described in the *RealTalk* paper to extract dyadic multimodal conversations from platforms such as YouTube, Spotify, and Dailymotion. This work is part of a broader research initiative aimed at building listening agents. The resulting dataset will also be reusable for other projects related to social interaction and multimodal AI.

## Setup Instructions

For detailed setup instructions, please refer to the ***[Setup Guide](setup.md)*** .

## Running step

### Running the Pipeline

1. **Complete the Installation**  
    Make sure you have followed all the steps in the [Setup Guide](setup.md) before proceeding.

2. **Configure and Run the Pipeline**  
    Open `pipeline_W10.py` and update the settings to match your environment and requirements.  
    Then, run the script to start the data extraction process.

3. **Clean and Merge the Data**  
    After the pipeline finishes, run `DataSetCleaningTools/dataclean.py`.  
    This script will check the integrity of the collected data and merge it into `.npy` files based on your chosen configuration.

## Pipeline Schema

![Pipeline Schema](img/schemaDMDC.png)

## Step-by-Step Pipeline Overview

1. **YouTube Video List**  
    Prepare a list of YouTube video URLs to process.

2. **Download Videos**  
    Download the videos from the provided URLs.

3. **Extract Audio**  
    Separate the audio track from each video.

4. **Segment Videos (Face Detection)**  
    Detect and segment faces in the video frames.

5. **Audio Diarization (Pyannote + Reclustering)**  
    Identify and separate different speakers in the audio.

6. **Assign Speakers to Segments**  
    Match detected faces to diarized speakers.

7. **Split Video by Segments (Dyadic Clips)**  
    Divide videos into clips containing two participants.

8. **Action Unit Extraction (OpenFace)**  
    Extract facial action units using OpenFace.

9. **Speaker Mapping (OpenFace + Diarization)**  
    Map facial features to corresponding speakers.

9.5 **Format AU with Speaker-Listener**  
    Structure action unit data by speaker and listener roles.

10. **Face Cropping (Optional)**  
     Crop faces from video frames if needed.

11. **Transcription (Parakeet or Whisper)**  
     Transcribe the audio to text.

12. **Format AU with n Frames (Windowing)**  
     Organize action unit data into fixed-length windows.

12.5 **Word Transcription (Align Words to AU Windows)**  
     Align transcribed words with action unit windows.

13. **Split Audio from Segments (Windowed)**  
     Extract windowed audio segments.

14. **MFCC Extraction (Speaker-Level Features)**  
     Compute MFCC features for each speaker.

15. **End of the Pipeline / Save Statistics**  
     Finalize processing and save summary statistics.