from whoIsSpeaking import run_diarization, assign_speakers_to_segments_from_df, merge_contiguous_segments, filter_short_segments
from extractMFCC import extractAndSaveMFCC
from OpenFace.actionUnitExtractVideo import process_FaceLandmarkVidMulti_from_container
from OpenFace.actionUnitForAVideo import process_FaceLandMark_video, process_AU_for_segments, extract_openface_features, run_openface_on_all_clips, detect_who_speaking_from_clips
from Filtering.filter import download_youtube_video, detect_faces_in_video, load_segments_from_csv, export_segments_with_speaker_to_csv, extract_audio_to_wav, split_audio_from_csv, wait_for_file_release
from formatAUSpeakerListener import format_all_clips
from SplitAudioVideo.splitVideoAndAudioFromSegment import splitVideoAndAudioFromSegment
from Whisper.transcriptFromAudio import transcriptFromAudio
from MFCCmergeWithDF import MFCCmergeWithDF
import librosa
import os
import warnings
import pandas as pd
import glob
import shutil
import subprocess
import platform

#Warnings deletes
warnings.filterwarnings("ignore", message="std\(\): degrees of freedom is <= 0")
warnings.filterwarnings("ignore", message=".*speechbrain.pretrained.*was deprecated.*")     


# Constants 

DATASET_FOLDER="V0.5DataSet"

def get_diarization_csv(wav_path):
    base_filename = os.path.splitext(os.path.basename(wav_path))[0]  # e.g. "1_video"
    diarization_folder = os.path.join(DATASET_FOLDER, "Diarization_Results")

    # Search for the CSV file that starts with the correct name
    pattern = os.path.join(diarization_folder, f"{base_filename}_diarization_results_*.csv")
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
    else:
        raise FileNotFoundError(f"No CSV file found for {base_filename} in {diarization_folder}")

def main_batch(video_list_file='videoV0.5.txt'):
    """
    Processes a batch of videos listed in a text file, performing a series of operations 
    including downloading, segmentation, audio extraction, diarization, speaker assignment, 
    feature extraction, transcription, and facial action unit analysis.
    Args:
        video_list_file (str): Path to the text file containing video URLs, one per line.
    Pipeline:
       
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

    print(f"\n\n\n---Step: 1--- Open video")
    with open(video_list_file, 'r') as f:
        video_urls = [line.strip() for line in f if line.strip()]
    

    for idx, url in enumerate(video_urls, start=1):
        try:
            output_name = os.path.join(os.path.dirname(__file__), DATASET_FOLDER, "mp4", f"{idx}_video.mp4")
            segments_csv = os.path.join(os.path.dirname(__file__), DATASET_FOLDER, "segments", f"{idx}_segments.csv")
            wav_dir = os.path.join(os.path.dirname(__file__), DATASET_FOLDER, "wav", f"{idx}_video")

            #####################
            # Download
            #####################
            print(f"\n\n\n ---Step: 2--- Dowloading")
            print(f"\n\n\n [Youtube] Downloading the video n°{idx} : {url}")
            download_youtube_video(url, output_name)

            #####################
            # Extract the Audio
            #####################
            print(f"\n\n\n ---Step: 3--- Extract the audio")

            extract_audio_to_wav(output_name, wav_dir)

            #####################
            # Segmentation (if needed)
            #####################
            print(f"\n\n\n ---Step: 4--- Segmentations")
            if os.path.exists(segments_csv):
                print("[Info] Segments already done, load segments from CSV ...")
                segments = load_segments_from_csv(segments_csv)
            else:
                print("[Info] Segments under creation with face detections...")
                segments = detect_faces_in_video(output_name)
            
            


            # Skip Diarization and Assignment if Segments Exist
            if os.path.exists(segments_csv):
                print("[Info] Segments already exist, skipping diarization and assignment...")
            else :
                #####################
                # Diarization (Identify which person is speaking)
                #####################
                print(f"\n\n\n ---Step: 5--- Diarization")
                print("[Diarization] Diarization starts")
                run_diarization(wav_dir)
                print("[Diarization] Diarization completed")

                csv_path = get_diarization_csv(output_name.replace(".mp4", ".wav"))
                df_diarization = pd.read_csv(csv_path)


                #####################
                # Identify which speaker is speaking during each segment using diarization results
                #####################
                print(f"\n\n\n ---Step: 6--- Assigning speakers to segments")
                print("[Assignment] Assigning speakers to segments...")

                merged = assign_speakers_to_segments_from_df(segments, df_diarization)
                merged = merge_contiguous_segments(merged, max_gap=1)
                merged = filter_short_segments(merged, min_duration=1.5)

                print("[Assignment] Assignment completed")
                export_segments_with_speaker_to_csv(merged, segments_csv)
                print("[Assignment] Assignment Exported")

            

            #####################
            # Split Audio/Video from segments 
            #####################
            print(f"\n\n\n ---Step: 7--- Split Audio/Video from segments")
            # Check if audio and video clips have already been processed
            clips_video_done = os.path.exists(os.path.join(os.path.dirname(__file__), DATASET_FOLDER, "clips_audio", f"{idx}_video.mp4"))

            output_audio = os.path.join(os.path.dirname(__file__), DATASET_FOLDER, "clips_audio")
            output_video = os.path.join(os.path.dirname(__file__), DATASET_FOLDER, "clips_video")

            if not clips_video_done:
                print(f"[Info] Splitting audio and video for video {idx}")
                input_path_csv = os.path.join(os.path.dirname(__file__), DATASET_FOLDER, "segments", f"{idx}_segments.csv")
                input_path_mp4 = os.path.join(os.path.dirname(__file__), DATASET_FOLDER, "mp4", f"{idx}_video.mp4")
                dyadicDF=splitVideoAndAudioFromSegment(os.path.abspath(input_path_mp4), os.path.abspath(input_path_csv), os.path.abspath(output_audio), os.path.abspath(output_video))
            else:
                print(f"[Info] Audio and video clips already processed for video {idx}, skipping...")

            
            #####################
            # MFCC Extract 
            #####################
            print(f"\n\n\n ---Step: 8--- Speaker Melspec")

            # Process MFCC for all audio clips in the directory and save as CSV
            print(f"[MFCC] Extracting MFCC features for audio clips in {output_audio}")
            mfcc_output_dir = os.path.join(os.path.dirname(__file__), DATASET_FOLDER, "mfcc_output", str(idx) + "_video")
            os.makedirs(mfcc_output_dir, exist_ok=True)
            audioFileFolder = os.path.join(output_audio, f"{idx}_video.mp4")
            audio_files = [file for file in glob.glob(os.path.join(audioFileFolder, "*.wav")) if os.path.isfile(file)]
            
            for audio_file in audio_files:
                mfcc_csv_path = os.path.join(mfcc_output_dir, f"{os.path.basename(audio_file).replace('.wav', '_mfcc.csv')}")
                if os.path.exists(mfcc_csv_path):
                    print(f"[MFCC] MFCC features already exist for {audio_file}, skipping...")
                    continue
                
                try:
                    print(f"[MFCC] Processing {audio_file}")
                    y, sr = librosa.load(audio_file, sr=None)
                    mfcc_features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128, hop_length=256).T
                    pd.DataFrame(mfcc_features).to_csv(mfcc_csv_path, index=False)
                    print(f"[MFCC] Saved MFCC features to {mfcc_csv_path}")
                except Exception as e:
                    print(f"[MFCC/ERR] Error processing {audio_file}: {e}")
                



            #####################
            # Openface Action Unit
            #####################
            print(f"\n\n\n ---Step: 9--- Action unit extraction")


            print(f"[AU] Processing AU")
            output = os.path.join(os.path.dirname(__file__), DATASET_FOLDER, 'AU_output', f'{idx}_video')
            if not os.path.exists(output):
                os.makedirs(output, exist_ok=True)
            clips_dir = DATASET_FOLDER+f"/clips_video/{idx}_video.mp4"
            openface_out_dir = DATASET_FOLDER+f"/openface_clips/{idx}_video"
            os.makedirs(openface_out_dir, exist_ok=True)
            run_openface_on_all_clips(clips_dir, openface_out_dir)

            # input_path_mp4 = os.path.join(os.path.dirname(__file__), DATASET_FOLDER, "mp4", f"{idx}_video.mp4")
            # output_path = os.path.join(os.path.dirname(__file__), DATASET_FOLDER,"openface_clips")
            # os.makedirs(output_path, exist_ok=True)

            # # openface_out_dir = DATASET_FOLDER+f"/openface_clips/{idx}_video"
            # process_FaceLandmarkVidMulti_from_container(input_path_mp4,output_path)
            # print(f"[AU] AU processing done")

            #################
            # Who is speaking
            #################
            print(f"\n\n\n ---Step: 10--- Who is speaking")
            print(f"[WhoIsSpeaking] Detecting who is speaking in video {idx}")
            detect_who_speaking_from_clips(
                video_id=str(idx),
                segments_csv_path=DATASET_FOLDER+f"/segments/{idx}_segments.csv",
                openface_dir=openface_out_dir,
                output_path=DATASET_FOLDER+f"/mapping_results/{idx}_video/mapping.csv"
            )
            print(f"[WhoIsSpeaking] Who is speaking completed for video {idx}")

            #####################
            # Format AU with Speaker-Listener
            #####################
            print(f"\n\n\n ---Step: 11--- Format AU with Speaker-Listener")
            format_all_clips(
                mapping_csv=DATASET_FOLDER+f"/mapping_results/{idx}_video/mapping.csv",
                openface_dir=DATASET_FOLDER+f"/openface_clips/{idx}_video",
                output_dir=DATASET_FOLDER+f"/formatted_clips/{idx}_video"
            )



            #####################
            # End of the pipe, delete cache
            #####################
            # if os.path.exists(output_name):
            #     os.remove(output_name)
            #     print(f"[Main/Info] Video deleted: {output_name}")
        except Exception as e:
            print(f"[Main/ERR] Error while processing the video {url} : {e}")

if __name__ == "__main__":
    main_batch()