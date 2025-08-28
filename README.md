# DMDC : Dyadic Multimodal Dataset Construction
## Project Description

The goal of this project is to reproduce the data creation process described in the *RealTalk* paper to extract dyadic multimodal conversations from platforms such as YouTube, Spotify, and Dailymotion. This work is part of a broader research initiative aimed at building listening agents. The resulting dataset will also be reusable for other projects related to social interaction and multimodal AI.

### General overview of the pipeline
The original purpose of the pipeline is to construct a multimodal dataset from dyadic conversations. Its objective is to enable the training of a machine learning model capable of predicting emotions and human interactions as it's showned below.

![Global Pipeline Schema](img/globalschemaDMDC.png)


## Setup Instructions

For detailed setup instructions, please refer to the ***[Setup Guide](setup.md)*** .
> **Note:** The provided Setup Guide is tailored for the main pipeline. If you plan to add new modules or modify the pipeline, please ensure you manually add any additional libraries to the Conda environment. Success is not guaranteed for custom modifications.
> **Note:** This project is primarily designed and tested for Linux environments, ensuring the most straightforward installation and usage experience. While Windows is also supported, additional configuration or troubleshooting may be required.

## How to run the pipeline ?

1. **Complete the Installation**  
    Make sure you have followed all the steps in the *[Setup Guide](setup.md)* before proceeding.

2. **Configure and Run the Pipeline**  
    - Ensure you have activated the appropriate Conda environment.  
    - Edit the configuration settings in `pipeline_W10.py` to match your system paths and project requirements.  
    - Run the openface container
        ```bash
        docker run -it --rm algebr/openface:latest
        #OR
        sudo docker run -it --rm algebr/openface:latest
        ```
    - Run the pipeline using the Conda environment’s Python executable:  
      ```bash
      conda activate dmdc-pipeline
      python pipeline_W10.py
      ```
    This will initiate the data extraction process according to your specified configuration.

3. **Clean and Merge the Data**  
    After the pipeline finishes, run `DataSetCleaningTools/dataclean.py`.  
    This script will check the integrity of the collected data and merge it into `.npy` files based on your chosen configuration.

## Pipeline Schema

![Pipeline Schema](img/schemaDMDC.png)
/!\ WIP to determine which one is better
![Pipeline Schema](img/schemaDMDCvert.png)


**Step-by-Step Pipeline Overview:**

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

    9.5. **Format AU with Speaker-Listener**  
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


## Additional Notes

If you encounter issues related to video downloading or extraction, ensure you have the lastest version of `yt_dlp` installed in your environment

```bash
pip install --upgrade yt-dlp
```


## Acknowledgments

Thank you for your interest in this project.  
Developed at the UMONS ISIA Lab under the supervision of Thomas B. and Kevin E., by Gaspard C. and Hugo M.

## Dependancies :

- **[OpenFace](https://github.com/TadasBaltrusaitis/OpenFace)** (Action units)

- **[Mediapipe](https://github.com/google-ai-edge/mediapipe)** (Type of scene)

- **[Nvidia-Parakeet](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)**(Transcription)

- **[Pyannote](https://github.com/pyannote/pyannote-audio)**(Diarization)

- **[Whisper](https://github.com/openai/whisper)**(Transcription)