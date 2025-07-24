from Tools.filter import detect_faces_in_video
from Tools.filter import download_youtube_video_480p_h264, load_segments_from_csv, export_segments_with_speaker_to_csv, extract_audio_to_wav, extract_dyadic_clips
from Transcribe.basic_transcribe import *
import os
import warnings
import subprocess
import platform
import time
import pandas as pd

#Warnings deletes
warnings.filterwarnings("ignore", message="std\(\): degrees of freedom is <= 0")
warnings.filterwarnings("ignore", message=".*speechbrain.pretrained.*was deprecated.*")     



# Constants 

DATASET_FOLDER="V0.11DataSet"
VIDEO_TEXT_FILE="./VideoList/videos_benchmarkDMDCW11.txt"
WINDOWING_SIZE_FRAME=128
WINDOWING_SIZE_STEP=32
START_VIDEO=22
END_VIDEO=24








def run_pipeline(video_list_file='videoV0.5.txt'):
    """
    
    """
    
    all_stat=[]
    def print_step_box(step_num, step_title):
        box_width = 80
        top = f"\n{'═'*box_width}"
        middle = f"║{f'{step_num}. {step_title}'.center(box_width-2)}║"
        bottom = f"{'═'*box_width}\n"
        print(top)
        print(middle)
        print(bottom)

    # ╔════════════════════════════════════════════════════════════════════════╗
    # ║                           1  Open Videos                               ║
    # ╚════════════════════════════════════════════════════════════════════════╝

    print_step_box(1, "Open Videos")

    with open(video_list_file, 'r') as f:
        video_urls = [line.strip() for line in f if line.strip()]


    # Ensure DATASET_FOLDER and all required subfolders exist (centralized)
        base_dir = os.path.join(os.path.dirname(__file__), DATASET_FOLDER)
        required_dirs = [
            base_dir,
            os.path.join(base_dir, "mp4"),
            os.path.join(base_dir, "wav"),
            os.path.join(base_dir, "transcripts"),
            os.path.join(base_dir, "segments"),
        ]
        for d in required_dirs:
            os.makedirs(d, exist_ok=True)


    for idx, url in enumerate(video_urls[START_VIDEO-1:END_VIDEO], start=START_VIDEO):
        try:

            stat_one_vid={}

            output_name = os.path.join(os.path.dirname(__file__), DATASET_FOLDER, "mp4", f"{idx}_video.mp4")
            segments_csv = os.path.join(os.path.dirname(__file__), DATASET_FOLDER, "segments", f"{idx}_segments.tsv")
            wav_dir = os.path.join(os.path.dirname(__file__), DATASET_FOLDER, "wav", f"{idx}_video")

            # ╔════════════════════════════════════════════════════════════════════════╗
            # ║                           2  Downloading                               ║
            # ╚════════════════════════════════════════════════════════════════════════╝
            print_step_box(2, "Downloading")
            start_time_process = time.time()


            
            print(f"\n\n\n [Youtube] Downloading the video n°{idx} : {url}")
            fps_video=round(download_youtube_video_480p_h264(url, output_name))
            print("[Info] Fps :",fps_video)


            stat_one_vid["2.Download"]=time.time()-start_time_process
            # ╔════════════════════════════════════════════════════════════════════════╗
            # ║                           3  Extract the audio                         ║
            # ╚════════════════════════════════════════════════════════════════════════╝

            print_step_box(3, "Extract the audio")


            start_time_process = time.time()

            extract_audio_to_wav(output_name, wav_dir)


            stat_one_vid["3.AudioExtract"]=time.time()-start_time_process

            # ╔════════════════════════════════════════════════════════════════════════╗
            # ║                           4  Segmentation                              ║
            # ╚════════════════════════════════════════════════════════════════════════╝
            print_step_box(4, "Segmentation")

            start_time_process = time.time()



            if os.path.exists(segments_csv):
                print("[Info] Segments already done, load segments from CSV ...")
            else:
                print("[Info] Segments under creation with face detections...")
                segments = detect_faces_in_video(output_name)
                # Save segments as TSV using pandas
                tsv_path = os.path.join(os.path.dirname(__file__), DATASET_FOLDER, "segments", f"{idx}_segments.tsv")
                df = pd.DataFrame(segments)
                # Save all existing columns to TSV
                df.to_csv(tsv_path, sep="\t", index=False)
                stat_one_vid["4.Extract"]=time.time()-start_time_process

            
            # ╔════════════════════════════════════════════════════════════════════════╗
            # ║                         14 Transcription                               ║
            # ╚════════════════════════════════════════════════════════════════════════╝
            print_step_box(14, "Transcription")
            start_time_process = time.time()

           


            # Whisper (Transcript)
            print(f"[Transcription] Transcribing audio segments for video using Whisper")
            audio_file = os.path.join(base_dir, DATASET_FOLDER, "wav", f"{idx}_video")
            transcript_output_folder = os.path.join(base_dir, DATASET_FOLDER, "transcripts", f"{idx}_video")
            os.makedirs(transcript_output_folder, exist_ok=True)

            tsv_output_path = os.path.join(transcript_output_folder, f"{idx}_transcribe.tsv")
            
            model = load_whisper_model("medium.en")
            segments = transcribe_audio_with_timestamps(model, audio_file)
            save_segments_to_tsv(segments, tsv_output_path)

            print(f"Transcription with timestamps saved to {tsv_path}")
            print(f"[Transcription] Transcription completed for video {idx}")
            stat_one_vid["14.Transcription"] = time.time() - start_time_process



        except Exception as e:
            print(f"[Main/ERR] Error while processing the video {url} : {e}")



if __name__ == "__main__":
    # run_pipeline("./VideoList/videoV0.5test.txt")
    # run_pipeline("./VideoList/videoV0.5.txt")
    run_pipeline(VIDEO_TEXT_FILE)