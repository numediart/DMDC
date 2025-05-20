from whoIsSpeaking import run_diarization, assign_speakers_to_segments_from_df
from extractMFCC import extractAndSaveMFCC
from OpenFace.actionUnitForAVideo import process_FaceLandMark_video
from Filtering.filter import download_youtube_video, detect_faces_in_video, load_segments_from_csv, export_segments_with_speaker_to_csv, extract_audio_to_wav
import librosa
import os
import warnings
import pandas as pd
import glob

# Masquer warning PyTorch sur std()
warnings.filterwarnings("ignore", message="std\(\): degrees of freedom is <= 0")

# Masquer warning SpeechBrain redirection
warnings.filterwarnings("ignore", message=".*speechbrain.pretrained.*was deprecated.*")

def get_diarization_csv(wav_path):
    base_filename = os.path.splitext(os.path.basename(wav_path))[0]  # e.g. "1_video"
    diarization_folder = os.path.join("V0DataSet", "Diarization_Results")

    # Cherche le fichier CSV qui commence par le bon nom
    pattern = os.path.join(diarization_folder, f"{base_filename}_diarization_results_*.csv")
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
    else:
        raise FileNotFoundError(f"Aucun fichier CSV trouvé pour {base_filename} dans {diarization_folder}")

def main_batch(video_list_file='videoV0.txt'):
    with open(video_list_file, 'r') as f:
        video_urls = [line.strip() for line in f if line.strip()]
    
    for idx, url in enumerate(video_urls, start=1):
        try:
            output_name = f"V0DataSet/mp4/{idx}_video.mp4"
            segments_csv = f"V0DataSet/segments/{idx}_segments.csv"
            wav_dir = f"V0DataSet/wav/{idx}_video"

            print(f"\n--- Traitement de la vidéo {idx}: {url} ---")
            download_youtube_video(url, output_name)

            if os.path.exists(segments_csv):
                print("Segments déjà détectés, chargement depuis CSV...")
                segments = load_segments_from_csv(segments_csv)
            else:
                print("Détection des visages en cours...")
                segments = detect_faces_in_video(output_name)
            
            extract_audio_to_wav(output_name, wav_dir)

            run_diarization(wav_dir)

            print("Diarization terminée")

            csv_path = get_diarization_csv(output_name.replace(".mp4", ".wav"))
            df_diarization = pd.read_csv(csv_path)

            print("Assignation des speakers aux segments...")

            merged = assign_speakers_to_segments_from_df(segments, df_diarization)

            print("Assignation terminée")

            export_segments_with_speaker_to_csv(merged, segments_csv)

            mfcc_output_dir = f'V0DataSet/mfcc'
            os.makedirs(mfcc_output_dir, exist_ok=True)

            # MFCC
            signal, sr = librosa.load(wav_dir, sr=None)
            audio_name = os.path.splitext(os.path.basename(wav_dir))[0]
            extractAndSaveMFCC(signal, mfcc_output_dir, audio_name)

            print(f"Traitement AU")
            output = os.path.join('V0DataSet/output', f'{idx}_video')
            os.makedirs(output, exist_ok=True)
            process_FaceLandMark_video(output_name,output,seconds=0.5 )

            if os.path.exists(output_name):
                os.remove(output_name)
                print(f"Vidéo supprimée : {output_name}")

        except Exception as e:
            print(f"Erreur lors du traitement de la vidéo {url} : {e}")

if __name__ == "__main__":
    main_batch()