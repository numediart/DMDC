from PyannoteDiarizationV31.whoIsSpeaking import run_diarization, assign_speakers_to_segments_from_df, merge_contiguous_segments, filter_short_segments
# from OpenFace.actionUnitForAVideo imkport process_FaceLandMark_video, process_AU_for_segments, extract_openface_features, run_openface_on_all_clips, detect_who_speaking_from_clips
from Tools.filter import download_youtube_video, detect_faces_in_video, load_segments_from_csv, export_segments_with_speaker_to_csv, extract_audio_to_wav, split_audio_from_csv, wait_for_file_release, extract_dyadic_clips
from Whisper.transcriptFromAudio import transcriptFromAudio
from Tools.MFCCmergeWithDF import MFCCmergeWithDF
from Tools.formatAUSpeakerListener import format_all_clips
from PyannoteRecluster.pyannote_reclustering import recluster_pyannote_diarization
import librosa
import os
import warnings
import pandas as pd
import glob
import shutil
import subprocess
import platform
import sys
import time
from pathlib import Path

#Warnings deletes
warnings.filterwarnings("ignore", message="std\(\): degrees of freedom is <= 0")
warnings.filterwarnings("ignore", message=".*speechbrain.pretrained.*was deprecated.*")     

def get_diarization_csv(wav_path):
    base_filename = os.path.splitext(os.path.basename(wav_path))[0]  # e.g. "1_video"
    diarization_folder = os.path.join("V0DataSet", "Diarization_Results")

    # Search for the CSV file that starts with the correct name
    pattern = os.path.join(diarization_folder, f"{base_filename}_diarization_results.csv")
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
    else:
        raise FileNotFoundError(f"No CSV file found for {base_filename} in {diarization_folder}")

def main_batch(video_list_file='VideoList/videoV0.txt'):
    """
    Processes a batch of videos listed in a text file, performing a series of operations 
    including downloading, segmentation, audio extraction, diarization, speaker assignment, 
    feature extraction, transcription, and facial action unit analysis.
    Args:
        video_list_file (str): Path to the text file containing video URLs, one per line.
    Pipeline:
        1. **Video Download**:
           - Downloads each video from the provided URL and saves it as an MP4 file.
        2. **Segmentation**:
           - If a segments CSV file exists, loads the segments from it.
           - Otherwise, performs face detection to create segments.
        3. **Audio Extraction**:
           - Extracts audio from the video and saves it as WAV files.
        4. **Diarization**:
           - If segments already exist, skips diarization.
           - Otherwise, performs speaker diarization to identify speakers in the audio.
        5. **Speaker Assignment**:
           - Assigns speakers to the detected segments using diarization results.
           - Exports the segments with speaker information to a CSV file.
        6. **MFCC Extraction**:
           - Extracts Mel-Frequency Cepstral Coefficients (MFCC) from the audio signal 
             and saves them for further analysis.
        7. **Transcription**:
           - Uses the Whisper model to transcribe the audio into text and saves the transcript.
        8. **Facial Action Unit Analysis**:
           - Processes the video to extract facial action units (AUs) using OpenFace.
        9. **Cleanup**:
           - Deletes the downloaded video file to free up storage.
    Exceptions:
        - Catches and logs any errors encountered during the processing of each video.
    Note:
        Ensure all required dependencies and external tools (e.g., Whisper, OpenFace) 
        are properly installed and configured before running this function.
        Check : requirement.txt
    """


    ###################
    # Open videos
    ###################
    with open(video_list_file, 'r') as f:
        video_urls = [line.strip() for line in f if line.strip()]
    

    for idx, url in enumerate(video_urls, start=13):
        try:
            output_name = f"V0DataSet/mp4/{idx}_video.mp4"
            segments_csv = f"V0DataSet/segments/{idx}_segments.csv"
            wav_dir = f"V0DataSet/wav/{idx}_video.wav"         

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
            
            
            # #####################
            # # Extract the Audio
            # #####################
            print(f"[Audio] Extracting audio from video...")
            extract_audio_to_wav(output_name, wav_dir)

            # Skip Diarization and Assignment if Segments Exist
            if os.path.exists(segments_csv):
                print("[Info] Segments already exist, skipping diarization and assignment...")
                csv_path = get_diarization_csv(output_name.replace(".mp4", ".wav"))
                df_diarization = pd.read_csv(csv_path)
            else :
            #     #####################
            #     # Diarization (Identify which person is speaking)
            #     #####################
                start_time = time.time()
                print("[Diarization] Diarization starts")
                run_diarization(wav_dir, "V0DataSet")
                print("[Diarization] Diarization completed")

                csv_path = get_diarization_csv(output_name.replace(".mp4", ".wav"))
                df_diarization = pd.read_csv(csv_path)

                print(f"[Diarization] Starting reclustering")
                recluster_pyannote_diarization(wav_dir, csv_path, f"V0DataSet/Diarization_Results/{idx}_video_diarization_results.csv", num_speakers=2)
                print(f"[Diarization] Reclustering completed")
                end_time = time.time()
                print(f"[Diarization] Time taken for diarization: {end_time - start_time:.2f} seconds")

            #####################
            # Identify which speaker is speaking during each segment using diarization results
            #####################
            print("[Assignment] Assigning speakers to segments...")

            merged = assign_speakers_to_segments_from_df(segments, df_diarization)
            print("[Assignment] merged 1")
            merged = merge_contiguous_segments(merged, max_gap=1)
            print("[Assignment] merged 2")
            merged = filter_short_segments(merged, min_duration=1.5)

            print("[Assignment] Assignment completed")
            export_segments_with_speaker_to_csv(merged, segments_csv)
            print("[Assignment] Assignment Exported")

             


            #####################
            # MFCC Extract 
            #####################
            # mfcc_output_dir = f'V0DataSet/mfcc'
            # os.makedirs(mfcc_output_dir, exist_ok=True)
            # signal, sr = librosa.load(wav_dir, sr=None)
            # audio_name = os.path.splitext(os.path.basename(wav_dir))[0]
            # mfcc=extractAndSaveMFCC(signal, mfcc_output_dir, audio_name)
            # df_mfcc=pd.DataFrame(mfcc)
            # df_mfcc.to_csv(os.path.join(mfcc_output_dir, f"{audio_name}_mfcc.csv"), index=False)


            # print("[MFCC] MFCC extraction done")

            # print("[MFCC] MFCC DF merge in progress")
            # mfccExportPath=os.path.join(os.path.dirname(__file__),"V0DataSet/segments/")
            # segments_mfcc_csv= pd.read_csv("./V0DataSet/segments/1_segments.csv")
            # mfcc= pd.read_csv("./V0DataSet/mfcc/1_video_mfcc.csv")
            # mfccExportPath=os.path.join(os.path.dirname(__file__),"V0DataSet/segments/","1_segments.csv")
            # MFCCmergeWithDF(segments_mfcc_csv,mfcc,mfccExportPath)
            # print("######[MFCC/DEBUG]######",segments_mfcc_csv,mfccExportPath)
            # MFCCmergeWithDF(segments_mfcc_csv,mfcc,mfccExportPath)
            # print("[MFCC] MFCC DF merge done")

            # #####################
            # # Path for transcription
            # #####################

            base_dir = os.path.dirname(__file__)
            output_tmp_wav = os.path.join(base_dir, "V0DataSet", "tmp_wav")
            if platform.system() == "Windows":
                python_path = os.path.join(".venv_parakeet", "Scripts", "python.exe")
            else:
                python_path = os.path.join(".venv_parakeet", "bin", "python")

            #####################
            # Splitting WAV from timestamps
            #####################
            output_tmp_wav = os.path.join(os.path.dirname(__file__), "V0DataSet", "tmp_wav", f"{idx}_video.wav")
            segment_paths = split_audio_from_csv(wav_dir, segments_csv, output_tmp_wav)

            # #####################
            # # Parakeet (Transcript) 
            # #####################
            print(f"[Transcription] Transcribing audio segments for video using Parakeet")
            subprocess.run([
                python_path,
                os.path.join(base_dir, "Transcribe", "transcribe_parakeet.py"),
                str(idx),
                *segment_paths
            ])
            print(f"[Transcription] Transcription completed for video {idx}")

            if os.path.exists(output_tmp_wav):
                if wait_for_file_release(output_tmp_wav):
                    os.remove(output_tmp_wav)
                else:
                    print(f"[WARN] Could not delete {output_tmp_wav} - file in use.")

            # transcriptFromAudio(audiofile=wav_dir, outputFolder=output_folder_whisper, modelType="tiny")

            # subprocess.run([
            #     sys.executable,
            #     "transcribe_parakeet.py",
            #     str(idx),
            #     *segment_paths
            # ])

            # #####################
            # # Split the video into dyadic clips
            # #####################
            # print(f"[Splitting] Processing dyadic clips for video {idx}")
            # extract_dyadic_clips(str(idx))
            # print(f"[Splitting] Dyadic clips processing done")

            # #####################
            # # Openface Action Unit
            # #####################
            # print(f"[AU] Processing AU")
            # output = os.path.join(os.path.dirname(__file__),'V0DataSet/output', f'{idx}_video')
            # if not os.path.exists(output):
            #     os.makedirs(output, exist_ok=True)
            # clips_dir = f"V0DataSet/clips_dyadic/{idx}_video"
            # openface_out_dir = f"V0DataSet/openface_clips/{idx}_video"
            # os.makedirs(openface_out_dir, exist_ok=True)
            # run_openface_on_all_clips(clips_dir, openface_out_dir)
            # print(f"[AU] AU processing done")

            # #################
            # # Who is speaking
            # #################
            # print(f"[WhoIsSpeaking] Detecting who is speaking in video {idx}")
            # detect_who_speaking_from_clips(
            #     video_id=str(idx),
            #     segments_csv_path=f"V0DataSet/segments/{idx}_segments.csv",
            #     openface_dir=openface_out_dir,
            #     output_path=f"V0DataSet/mapping_results/{idx}_video/mapping.csv"
            # )
            # print(f"[WhoIsSpeaking] Who is speaking completed for video {idx}")

            # #####################
            # # Format AU with Speaker-Listener
            # #####################
            # format_all_clips(
            #     mapping_csv=f"V0DataSet/mapping_results/{idx}_video/mapping.csv",
            #     openface_dir=f"V0DataSet/openface_clips/{idx}_video",
            #     output_dir=f"V0DataSet/formatted_clips/{idx}_video"
            # )

            #####################
            # End of the pipe, delete cache
            #####################
            # if os.path.exists(output_name):
            #     os.remove(output_name)
            #     print(f"[Main/Info] Video deleted: {output_name}")
        except Exception as e:
            print(f"[Main/ERR] Error while processing the video {url} : {e}")

if __name__ == "__main__":
    main_batch("./VideoList/newVideoHugo.txt")