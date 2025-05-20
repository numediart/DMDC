from whoIsSpeaking import run_diarization, assign_speakers_to_segments_from_df
from extractMFCC import extractAndSaveMFCC
from OpenFace.actionUnitForAVideo import process_FaceLandMark_video
from Filtering.filter import download_youtube_video, detect_faces_in_video, load_segments_from_csv, export_segments_with_speaker_to_csv, extract_audio_to_wav
from Whisper.transcriptFromAudio import transcriptFromAudio
import librosa
import os
import warnings
import pandas as pd
import glob

#Warnings deletes
warnings.filterwarnings("ignore", message="std\(\): degrees of freedom is <= 0")
warnings.filterwarnings("ignore", message=".*speechbrain.pretrained.*was deprecated.*")



def get_diarization_csv(wav_path):
    base_filename = os.path.splitext(os.path.basename(wav_path))[0]  # e.g. "1_video"
    diarization_folder = os.path.join("V0DataSet", "Diarization_Results")

    # Search for the CSV file that starts with the correct name
    pattern = os.path.join(diarization_folder, f"{base_filename}_diarization_results_*.csv")
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
    else:
        raise FileNotFoundError(f"No CSV file found for {base_filename} in {diarization_folder}")

def main_batch(video_list_file='videoV0.txt'):

    ###################
    # Open videos
    ###################
    with open(video_list_file, 'r') as f:
        video_urls = [line.strip() for line in f if line.strip()]
    

    for idx, url in enumerate(video_urls, start=1):
        try:
            output_name = f"V0DataSet/mp4/{idx}_video.mp4"
            segments_csv = f"V0DataSet/segments/{idx}_segments.csv"
            wav_dir = f"V0DataSet/wav/{idx}_video"

            #####################
            # Download
            #####################
            print(f"\n\n\n[Youtube] Downloading the video n°{idx} : {url}")
            download_youtube_video(url, output_name)

            
            #####################
            # Segmentation (if needed)
            #####################
            if os.path.exists(segments_csv):
                print("[Info] Segments already done, load segments from CSV ...")
                segments = load_segments_from_csv(segments_csv)
            else:
                print("[Info] Segments under creation with face detections...")
                segments = detect_faces_in_video(output_name)
            
            
            #####################
            # Extract the Audio
            #####################
            extract_audio_to_wav(output_name, wav_dir)

            
            #####################
            # Diarization (Identify which person is speaking)
            #####################
            print("[Diarization] Diarization starts")
            run_diarization(wav_dir)
            print("[Diarization] Diarization completed")

            csv_path = get_diarization_csv(output_name.replace(".mp4", ".wav"))
            df_diarization = pd.read_csv(csv_path)


            #####################
            # Identify which speaker is speaking during each segment using diarization results
            #####################
            print("[Assignment] Assigning speakers to segments...")

            merged = assign_speakers_to_segments_from_df(segments, df_diarization)

            print("[Assignment] Assignment completed")
            export_segments_with_speaker_to_csv(merged, segments_csv)
            print("[Assignment] Assignment Exported")

            mfcc_output_dir = f'V0DataSet/mfcc'
            os.makedirs(mfcc_output_dir, exist_ok=True)


            #####################
            # MFCC Extract 
            #####################
            signal, sr = librosa.load(wav_dir, sr=None)
            audio_name = os.path.splitext(os.path.basename(wav_dir))[0]
            extractAndSaveMFCC(signal, mfcc_output_dir, audio_name)
            print("[MFCC] MFCC extraction done")


            #####################
            # Whisper (Transcript) 
            #####################
            output_folder_whisper=os.path.join("V0DataSet", "transcript")
            transcriptFromAudio(audiofile=wav_dir,outputFolder=output_folder_whisper,modelType="tiny")

            #####################
            # Openface Action Unit
            #####################
            print(f"[AU] Processing AU")
            output = os.path.join('V0DataSet/output', f'{idx}_video')
            os.makedirs(output, exist_ok=True)
            process_FaceLandMark_video(output_name,output,seconds=0.5 )


            #####################
            # End of the pipe, delete cache
            #####################
            if os.path.exists(output_name):
                os.remove(output_name)
                print(f"Vidéo supprimée : {output_name}")

        except Exception as e:
            print(f"Erreur lors du traitement de la vidéo {url} : {e}")

if __name__ == "__main__":
    main_batch()