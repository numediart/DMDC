from whoIsSpeaking import run_diarization
from Filtering.filter import process_video
from extractMFCC import extractAndSaveMFCC
from OpenFace.actionUnitForAVideo import process_FaceLandMark_video
import librosa
import os
import warnings

# Masquer warning PyTorch sur std()
warnings.filterwarnings("ignore", message="std\(\): degrees of freedom is <= 0")

# Masquer warning SpeechBrain redirection
warnings.filterwarnings("ignore", message=".*speechbrain.pretrained.*was deprecated.*")

def main_batch(video_list_file='videoV0.txt'):
    with open(video_list_file, 'r') as f:
        video_urls = [line.strip() for line in f if line.strip()]
    
    for idx, url in enumerate(video_urls, start=1):
        try:
            process_video(url, idx)

            # Dossier contenant les fichiers .wav extraits pour chaque clip
            clip_dir = f'V0DataSet/clips/{idx}_video'
            clip_wav_dir = f'V0DataSet/wav/{idx}_video'
            mfcc_output_dir = f'V0DataSet/mfcc/{idx}_video'
            os.makedirs(mfcc_output_dir, exist_ok=True)

            for file in os.listdir(clip_wav_dir):
                if file.endswith('.wav'):
                    wav_path = os.path.join(clip_wav_dir, file)
                    print(f"\nTraitement du fichier audio : {wav_path}")

                    # Diarization
                    run_diarization(wav_path)

                    print("Diarization terminée")

                    # MFCC
                    signal, sr = librosa.load(wav_path, sr=None)
                    output_base_name = os.path.splitext(file)[0]
                    extractAndSaveMFCC(signal, mfcc_output_dir, output_base_name)

            for file in os.listdir(clip_dir):
                if file.endswith('.mp4'):
                    clip_path = os.path.join(clip_dir, file)
                    print(f"Traitement AU du clip : {clip_path}")

                    clip_name = os.path.splitext(file)[0]  # ex: 'clip_001'
                    output = os.path.join('V0DataSet/output', f'{idx}_video', clip_name)
                    os.makedirs(output, exist_ok=True)

                    process_FaceLandMark_video(clip_path,output,seconds=0.5 )

        except Exception as e:
            print(f"Erreur lors du traitement de la vidéo {url} : {e}")

if __name__ == "__main__":
    main_batch()