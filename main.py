from whoIsSpeaking import run_diarization
from Filtering.filter import process_video

def main_batch(video_list_file='videoV0.txt'):
    with open(video_list_file, 'r') as f:
        video_urls = [line.strip() for line in f if line.strip()]
    
    for idx, url in enumerate(video_urls, start=1):
        try:
            process_video(url, idx)
            run_diarization(f"{idx}_video.wav")
        except Exception as e:
            print(f"Erreur lors du traitement de la vidéo {url} : {e}")

if __name__ == "__main__":
    main_batch()