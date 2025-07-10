from PyannoteDiarizationV31.whoIsSpeaking import run_diarization, assign_speakers_to_segments_from_df, merge_contiguous_segments, filter_short_segments
from OpenFace.actionUnitExtractVideo import process_FaceLandmarkVidMulti_from_container
from PyannoteRecluster.pyannote_reclustering import recluster_pyannote_diarization
from OpenFace.actionUnitForAVideo import process_FaceLandMark_video, process_AU_for_segments, extract_openface_features, run_openface_on_all_clips, detect_who_speaking_from_clips
from Tools.filter import download_youtube_video, detect_faces_in_video, load_segments_from_csv, export_segments_with_speaker_to_csv, extract_audio_to_wav, split_audio_from_csv, wait_for_file_release, extract_dyadic_clips
from Tools.formatAUSpeakerListener import format_all_clips
from Tools.splitVideoAndAudioFromSegment import extract_audio_segment,extract_video_segments
from Whisper.transcriptFromAudio import transcriptFromAudio
from Tools.MFCCmergeWithDF import MFCCmergeWithDF
from Tools.get_diarization_csv import get_diarization_csv
from Tools.split_csv_with_sliding_window import split_csv_with_sliding_window
import librosa
import os
import warnings
import pandas as pd
import glob
import numpy as np
import shutil
import subprocess
import platform
import time

#Warnings deletes
warnings.filterwarnings("ignore", message="std\(\): degrees of freedom is <= 0")
warnings.filterwarnings("ignore", message=".*speechbrain.pretrained.*was deprecated.*")     



# Constants 

DATASET_FOLDER="V0.10DataSet"
VIDEO_TEXT_FILE="./VideoList/videoV0.txt"
WINDOWING_SIZE_FRAME=128








def run_pipeline(video_list_file='videoV0.5.txt'):
    """
    Processes a batch of videos listed in a text file, performing a series of operations 
    including downloading, segmentation, audio extraction, diarization, speaker assignment, 
    feature extraction, and facial action unit analysis.
    Args:
        video_list_file (str): Path to the text file containing video URLs, one per line.
    Pipeline:

        ╔════════════════════════════════════╗
        ║  1. YouTube Video List             ║
        ╚════════════════════════════════════╝
                        │
                        ▼
        ╔════════════════════════════════════╗
        ║  2. Download Videos                ║
        ╚════════════════════════════════════╝
                        │
                        ▼
        ╔════════════════════════════════════╗
        ║  3. Extract / Process Audio        ║
        ╚════════════════════════════════════╝
                        │
                        ▼
        ╔════════════════════════════════════╗
        ║  4. Segment Videos                 ║
        ║     ├─ Dyadic                      ║
        ║     ├─ Single                      ║
        ║     └─ Other                       ║
        ╚════════════════════════════════════╝
                        │
                        ▼
        ╔════════════════════════════════════╗
        ║  5. Audio Diarization              ║
        ║     (Detect speakers)              ║
        ╚════════════════════════════════════╝
                        │
                        ▼
        ╔════════════════════════════════════╗
        ║  6. Assign Speakers to Segments    ║
        ╚════════════════════════════════════╝
                        │
                        ▼
        ╔════════════════════════════════════╗
        ║  7. Split Video by Timestamps      ║
        ╚════════════════════════════════════╝
                        │
                        ▼
        ╔════════════════════════════════════╗
        ║  8. AU Extraction                  ║
        ║     (Action Units per speaker)     ║
        ╚════════════════════════════════════╝
                        │
                        ▼
        ╔════════════════════════════════════╗
        ║  9. Format AU (64 frames)          ║
        ║     (Mark speaker)                 ║
        ╚════════════════════════════════════╝
                        │
                        ▼
        ╔════════════════════════════════════╗
        ║ 10. Face Cropping                  ║
        ╚════════════════════════════════════╝
                        │
                        ▼
        ╔════════════════════════════════════╗
        ║ 11. AU Windowing                   ║
        ║     (n = 64 frames)                ║
        ╚════════════════════════════════════╝
                        │
                        ▼
        ╔════════════════════════════════════╗
        ║ 12. Audio Segmentation             ║
        ║     (Aligned with AU window)       ║
        ╚════════════════════════════════════╝
                        │
                        ▼
        ╔════════════════════════════════════╗
        ║ 13. MFCC Extraction                ║
        ║     (Speaker-level features)       ║
        ╚════════════════════════════════════╝
                        │
                        ▼
        ╔════════════════════════════════════╗
        ║ 14. Transcription                  ║
        ║     (Text per speaker segment)     ║
        ╚════════════════════════════════════╝
                        │
                        ▼
        ╔══════════════════════════════════════════════════════════════╗
        ║ 15. Dataset Creation                                         ║
        ╚══════════════════════════════════════════════════════════════╝




    Exceptions:
        - Catches and logs any errors encountered during the processing of each video.
    Note:
        Ensure all required dependencies and external tools (e.g., OpenFace) 
        are properly installed and configured before running this function.
        Check : requirement.txt
    """
    
    all_stat=[]
    # ╔════════════════════════════════════════════════════════════════════════╗
    # ║                           1  Open Videos                               ║
    # ╚════════════════════════════════════════════════════════════════════════╝

    print(f"\n\n\n---Step: 1--- Open video")
    with open(video_list_file, 'r') as f:
        video_urls = [line.strip() for line in f if line.strip()]
    

    for idx, url in enumerate(video_urls, start=1):
        try:

            stat_one_vid={}

            output_name = os.path.join(os.path.dirname(__file__), DATASET_FOLDER, "mp4", f"{idx}_video.mp4")
            segments_csv = os.path.join(os.path.dirname(__file__), DATASET_FOLDER, "segments", f"{idx}_segments.csv")
            wav_dir = os.path.join(os.path.dirname(__file__), DATASET_FOLDER, "wav", f"{idx}_video")

            # ╔════════════════════════════════════════════════════════════════════════╗
            # ║                           2  Downloading                               ║
            # ╚════════════════════════════════════════════════════════════════════════╝
            print(f"\n\n\n ---Step: 2--- Downloading")
            start_time_process = time.time()




            print(f"\n\n\n [Youtube] Downloading the video n°{idx} : {url}")
            download_youtube_video(url, output_name)




            stat_one_vid["2.Download"]=time.time()-start_time_process
            # ╔════════════════════════════════════════════════════════════════════════╗
            # ║                           3  Extract the audio                         ║
            # ╚════════════════════════════════════════════════════════════════════════╝
            print(f"\n\n\n ---Step: 3--- Extract the audio")
            start_time_process = time.time()



            extract_audio_to_wav(output_name, wav_dir)



            stat_one_vid["3.AudioExtract"]=time.time()-start_time_process

            
            # ╔════════════════════════════════════════════════════════════════════════╗
            # ║                           4  Segmentation                              ║
            # ╚════════════════════════════════════════════════════════════════════════╝
            print(f"\n\n\n ---Step: 4--- Segmentations")
            start_time_process = time.time()



            # if os.path.exists(segments_csv):
            #     print("[Info] Segments already done, load segments from CSV ...")
            #     segments = load_segments_from_csv(segments_csv)
            # else:
            #     print("[Info] Segments under creation with face detections...")
            #     segments = detect_faces_in_video(output_name)


            # WARNING Test to remove the mediapipe printing
            if os.path.exists(segments_csv):
                print("[Info] Segments already done, load segments from CSV ...")
                segments = load_segments_from_csv(segments_csv)
            else:
                print("[Info] Segments under creation with face detections...")
                process = subprocess.Popen(
                    ['python3.10', '-c', f'import Filtering.filter as ff; ff.detect_faces_in_video("{output_name}")'],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                process.wait()



            stat_one_vid["4.Extract"]=time.time()-start_time_process

            
            # Skip 5.Diarization and 6.Assignment if Segments Exist
            if os.path.exists(segments_csv):
                print("[Info] Segments already exist, skipping diarization and assignment...")
            else :
                # ╔════════════════════════════════════════════════════════════════════════╗
                # ║                           5  Diarization                               ║
                # ╚════════════════════════════════════════════════════════════════════════╝
                print(f"\n\n\n ---Step: 5--- Diarization")
                start_time_process = time.time()



                # Basic diarization 
                print("[Diarization] Diarization starts")
                run_diarization(wav_dir,DATASET_FOLDER)
                print("[Diarization] Diarization completed")

                csv_path = get_diarization_csv(output_name.replace(".mp4", ".wav"))
                df_diarization = pd.read_csv(csv_path)


                # Recluster
                print(f"[Diarization] Starting reclustering")
                recluster_pyannote_diarization(wav_dir, csv_path, f"{DATASET_FOLDER}/Diarization_Results/{idx}_video_diarization_results.csv", num_speakers=2)
                print(f"[Diarization] Reclustering completed")

                
                print("[Diarization] Diarization completed")




                stat_one_vid["5.Diarization"]=time.time()-start_time_process

                # ╔════════════════════════════════════════════════════════════════════════╗
                # ║                6  Assigning speakers to segments                       ║
                # ╚════════════════════════════════════════════════════════════════════════╝
                print(f"\n\n\n ---Step: 6--- Assigning speakers to segments")
                start_time_process = time.time()




                print("[Assignment] Assigning speakers to segments...")

                merged = assign_speakers_to_segments_from_df(segments, df_diarization)
                merged = merge_contiguous_segments(merged, max_gap=1)
                merged = filter_short_segments(merged, min_duration=1.5)

                print("[Assignment] Assignment completed")
                export_segments_with_speaker_to_csv(merged, segments_csv)
                print("[Assignment] Assignment Exported")




                stat_one_vid["6.Assigning"]=time.time()-start_time_process


            # ╔════════════════════════════════════════════════════════════════════════╗
            # ║                    7  Split Video from segments                        ║
            # ╚════════════════════════════════════════════════════════════════════════╝
            print(f"\n\n\n ---Step: 7--- Split Video from segments")
            start_time_process = time.time()




            # Check if video clips have already been processed
            clips_video_done = os.path.exists(os.path.join(os.path.dirname(__file__), DATASET_FOLDER, "clips_video", f"{idx}_video.mp4"))

            # output_video = os.path.join(os.path.dirname(__file__), DATASET_FOLDER, "clips_video")

            if not clips_video_done:
                # print(f"[Info] Splitting video for video {idx}")
                # input_path_csv = os.path.join(os.path.dirname(__file__), DATASET_FOLDER, "segments", f"{idx}_segments.csv")
                # input_path_mp4 = os.path.join(os.path.dirname(__file__), DATASET_FOLDER, "mp4", f"{idx}_video.mp4")
                # dyadicDF=extract_video_segments(os.path.abspath(input_path_mp4), os.path.abspath(input_path_csv), os.path.abspath(output_video))
                print(f"[Splitting] Processing dyadic clips for video {idx}")
                extract_dyadic_clips(str(idx),DATASET_FOLDER)
                print(f"[Splitting] Dyadic clips processing done")
            else:
                print(f"[Info] Video clips already processed for video {idx}, skipping...")

            

            stat_one_vid["7.Split_video"]=time.time()-start_time_process


            # ╔════════════════════════════════════════════════════════════════════════╗
            # ║                    8  Action unit extraction                           ║
            # ╚════════════════════════════════════════════════════════════════════════╝
            print(f"\n\n\n ---Step: 8--- Action unit extraction")
            start_time_process = time.time()




            output = os.path.join(os.path.dirname(__file__), DATASET_FOLDER, 'openface_clips', f'{idx}_video')
            if not os.path.exists(output):
                print(f"[AU] Processing AU")
                os.makedirs(output, exist_ok=True)
                clips_dir = DATASET_FOLDER+f"/clips_video/{idx}_video"
                openface_out_dir = DATASET_FOLDER+f"/openface_clips/{idx}_video"
                os.makedirs(openface_out_dir, exist_ok=True)
                run_openface_on_all_clips(clips_dir, openface_out_dir)
            else:
                print(f"[AU] AU already done")




            stat_one_vid["8.OpenFace"]=time.time()-start_time_process

            # ╔════════════════════════════════════════════════════════════════════════╗
            # ║          9 Speaker Mapping with OpenFace and Diarization               ║
            # ╚════════════════════════════════════════════════════════════════════════╝
            print(f"\n\n\n ---Step: 9--- Speaker Mapping with OpenFace and Diarization")
            start_time_process = time.time()



            print(f"[WhoIsSpeaking] Detecting who is speaking in video {idx}")
            mapping_csv_path = os.path.join(os.path.dirname(__file__), DATASET_FOLDER, "mapping_results", f"{idx}_video", "mapping.csv")

            
            if os.path.exists(mapping_csv_path):
                print(f"[WhoIsSpeaking] Mapping already exists for video {idx}, skipping...")
            else:
                detect_who_speaking_from_clips(
                    video_id=str(idx),
                    segments_csv_path=os.path.join(os.path.dirname(__file__), DATASET_FOLDER, "segments",f"{idx}_segments.csv"),
                    openface_dir=os.path.join(os.path.dirname(__file__), DATASET_FOLDER, "openface_clips", f"{idx}_video"),
                    output_path=mapping_csv_path
                )
                print(f"[WhoIsSpeaking] Who is speaking completed for video {idx}")


            stat_one_vid["9.WhoIsSpeaking"]=time.time()-start_time_process
            


            # ╔════════════════════════════════════════════════════════════════════════╗
            # ║                 10 Format AU with Speaker-Listener                     ║
            # ╚════════════════════════════════════════════════════════════════════════╝
            print(f"\n\n\n ---Step: 10--- Format AU with Speaker-Listener")
            start_time_process = time.time()

            formatted_dir = os.path.join(os.path.dirname(__file__), DATASET_FOLDER, "formatted_clips", f"{idx}_video")
            if os.path.exists(formatted_dir) and os.listdir(formatted_dir):
                print(f"[Format] Formatted clips already exist for video {idx}, skipping...")
            else:
                format_all_clips(
                    mapping_csv=os.path.join(os.path.dirname(__file__), DATASET_FOLDER, "mapping_results", f"{idx}_video", "mapping.csv"),
                    openface_dir=os.path.join(os.path.dirname(__file__), DATASET_FOLDER, "openface_clips", f"{idx}_video"),
                    output_dir=formatted_dir
                )


            stat_one_vid["10.FormatAU"]=time.time()-start_time_process
            # ╔════════════════════════════════════════════════════════════════════════╗
            # ║                 10 Format AU with Speaker-Listener                     ║
            # ╚════════════════════════════════════════════════════════════════════════╝
            print(f"\n\n\n ---Step: 11---Format AU with 64 frames")
            start_time_process = time.time()



            print("[64f] Process Speaker files")
            formatted_dir = os.path.join(os.path.dirname(__file__), DATASET_FOLDER, "formatted_clips", f"{idx}_video", "speaker")
            windowed_dir = os.path.join(os.path.dirname(__file__), DATASET_FOLDER, "64_frames_windowed_clips", f"{idx}_video", "speaker")
            os.makedirs(windowed_dir, exist_ok=True)
            csv_files = glob.glob(os.path.join(formatted_dir, "*.csv"))
            for csv_file in csv_files:
                base_name = os.path.splitext(os.path.basename(csv_file))[0]
                # Check if at least one windowed file for this csv exists
                already_done = any(f.startswith(base_name + "_") for f in os.listdir(windowed_dir))
                if already_done:
                    print(f"[64f] Windowed files already exist for {csv_file}, skipping...")
                    continue
                split_csv_with_sliding_window(csv_file, windowed_dir)

            print("[64f] Process listener files")
            formatted_dir = os.path.join(os.path.dirname(__file__), DATASET_FOLDER, "formatted_clips", f"{idx}_video", "listener")
            windowed_dir = os.path.join(os.path.dirname(__file__), DATASET_FOLDER, "64_frames_windowed_clips", f"{idx}_video", "listener")
            os.makedirs(windowed_dir, exist_ok=True)
            csv_files = glob.glob(os.path.join(formatted_dir, "*.csv"))
            for csv_file in csv_files:
                base_name = os.path.splitext(os.path.basename(csv_file))[0]
                already_done = any(f.startswith(base_name + "_") for f in os.listdir(windowed_dir))
                if already_done:
                    print(f"[64f] Windowed files already exist for {csv_file}, skipping...")
                    continue
                split_csv_with_sliding_window(csv_file, windowed_dir)

            stat_one_vid["11.FormatTO64f"]=time.time()-start_time_process
            #####################
            # Split Audio from segments 
            #####################
            print(f"\n\n\n ---Step: 12--- Split Audio from segments")
            start_time_process = time.time()


            # Check if audio and video clips have already been processed
            clips_audio_done = os.path.exists(os.path.join(os.path.dirname(__file__), DATASET_FOLDER, "clips_audio", f"{idx}_video"))

            output_audio = os.path.join(os.path.dirname(__file__), DATASET_FOLDER, "clips_audio", f"{idx}_video")
            os.makedirs(output_audio, exist_ok=True)

            if not clips_audio_done:
                print(f"[Info] Splitting audio for video {idx}")
                input_path_wav = os.path.join(os.path.dirname(__file__), DATASET_FOLDER, "wav", f"{idx}_video")
                csv_64frames_path= os.path.join(os.path.dirname(__file__),DATASET_FOLDER, "64_frames_windowed_clips", f"{idx}_video","speaker")
                for npy_file in glob.glob(os.path.join(csv_64frames_path, "*.npy")):
                    npy_file_base = os.path.basename(npy_file)
                    print(npy_file_base.split("_"))
                    start_frm = int(npy_file_base.split("_")[0])
                    end_frm =  int(npy_file_base.split("_")[2])
                    print(f"[Audio] Extracting audio segment from frame {start_frm} to {end_frm} for {npy_file_base}")
                    extract_audio_segment(input_path_wav, start_frm, end_frm, output_audio)

            else:
                print(f"[Info] Video clips already processed for video {idx}, skipping...")


            stat_one_vid["12.SplitAudio"]=time.time()-start_time_process
            #####################
            # MFCC Extract 
            #####################
            print(f"\n\n\n ---Step: 13--- Speaker Melspec")
            start_time_process = time.time()

            # Process MFCC for all audio clips in the directory and save as CSV
            mfcc_output_dir = os.path.join(os.path.dirname(__file__), DATASET_FOLDER, "mfcc_output", str(idx) + "_video")
            os.makedirs(mfcc_output_dir, exist_ok=True)
            print(f"[MFCC] Extracting MFCC features for audio clips in {output_audio}")
            audio_files = [file for file in glob.glob(os.path.join(output_audio, "*.wav")) if os.path.isfile(file)]
            
            for audio_file in audio_files:
                mfcc_csv_path = os.path.join(mfcc_output_dir, f"{os.path.basename(audio_file).replace('.wav', '_mfcc.csv')}")
      
                try:
                    print(f"[MFCC] Processing {audio_file}")
                    y, sr = librosa.load(audio_file, sr=None)
                    mfcc_features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128, hop_length=256).T
                    np.save(mfcc_csv_path.replace('_mfcc.csv', '_mfcc.npy'), mfcc_features)
                    print(f"[MFCC] Saved MFCC features to {mfcc_csv_path.replace('_mfcc.csv', '_mfcc.npy')}")
                except Exception as e:
                    print(f"[MFCC/ERR] Error processing {audio_file}: {e}")
                

            stat_one_vid["13.MFCC"]=time.time()-start_time_process
            #####################
            # End of the pipe, delete cache
            #####################
            print(stat_one_vid)
            # Format the values in stat_one_vid to 3 decimal places
            stat_one_vid = {key: round(value, 4) for key, value in stat_one_vid.items()}
            all_stat.append(stat_one_vid)
            # Save statistics to CSV
            print(f"[Main/Info] Saving statistics to CSV...")
            stats_df = pd.DataFrame(all_stat)
            stats_csv_path = os.path.join(os.path.dirname(__file__), DATASET_FOLDER, "monitoring", f"{idx}_video_processing_statistics.csv")
            stats_df.to_csv(stats_csv_path, index=False)
            print(f"[Main/Info] Statistics saved to {stats_csv_path}")
            
            # if os.path.exists(output_name):
            #     os.remove(output_name)
            #     print(f"[Main/Info] Video deleted: {output_name}")
        except Exception as e:
            print(f"[Main/ERR] Error while processing the video {url} : {e}")



if __name__ == "__main__":
    run_pipeline(VIDEO_TEXT_FILE)