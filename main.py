from whoIsSpeaking import run_diarization
from Filtering.filter import process_video
from extractMFCC import extractAndSaveMFCC
import librosa
import os

def main_batch(video_list_file='videoV0.txt'):
    with open(video_list_file, 'r') as f:
        video_urls = [line.strip() for line in f if line.strip()]
    
    for idx, url in enumerate(video_urls, start=1):
        try:
            process_video(url, idx)
            run_diarization(f"{idx}_video.wav")

            wav_path = os.path.abspath(f'./V0DataSet/wav/{idx}_video.wav')
            signal, sr = librosa.load(wav_path)
            os.makedirs(os.path.dirname("V0DataSet/mffc/"), exist_ok=True)
            extractAndSaveMFCC(signal, './V0DataSet/mffc', f'{idx}_video')
        except Exception as e:
            print(f"Erreur lors du traitement de la vidéo {url} : {e}")

if __name__ == "__main__":
    main_batch()