from nemo.collections.asr.models import SortformerEncLabelModel
import csv
import pandas as pd
import librosa
import soundfile as sf
import os

import os

TEMPFOLDER="./tempsegment"


diar_model = SortformerEncLabelModel.restore_from(restore_path="./NemoDiarization/model/diar_sortformer_4spk-v1.nemo", map_location='cuda', strict=False)




def diarization_nvidia_sortformer_process(audio_input, output_path, segmentation_minutes=5, overlap_duration_sec=60):

    audio_base_name = os.path.basename(audio_input)
    if not os.path.exists(TEMPFOLDER):
        os.mkdir(TEMPFOLDER)

    audio, sr = librosa.load(audio_input, sr=16000, mono=True)
    segment_length = segmentation_minutes * 60 * sr
    overlap_length = int(overlap_duration_sec * sr)
    segments = [audio[i:i+segment_length] for i in range(0, len(audio), segment_length - overlap_length)]

    global_speaker_map = {}
    next_speaker_id = 1
    results = []

    for idx, segment in enumerate(segments):
        seg_path = f"{TEMPFOLDER}/segment_{idx}_{audio_base_name}.wav"
        sf.write(seg_path, segment, sr)

        diar_data = diar_model.diarize(audio=seg_path, batch_size=1)[0]
        offset = (segment_length - overlap_length) * idx / sr

        for seg_line in diar_data:
            start, end, spk = seg_line.split()
            spk_clean = spk.strip(':').upper()
            if spk_clean not in global_speaker_map:
                global_speaker_map[spk_clean] = f"SPEAKER_{next_speaker_id}"
                next_speaker_id += 1
            results.append({
                "start_time": round(float(start) + offset, 2),
                "end_time": round(float(end) + offset, 2),
                "speaker": global_speaker_map[spk_clean]
            })

    for f in os.listdir(TEMPFOLDER):
        os.remove(os.path.join(TEMPFOLDER, f))
    os.rmdir(TEMPFOLDER)

    df = pd.DataFrame(results).sort_values("start_time").reset_index(drop=True)
    out_csv = f"{output_path}/{audio_base_name.replace('.wav', '')}_diarization.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved diarization to {out_csv}")


if __name__ == "__main__":
    output_path="./NemoDiarization/output/v0/"
    # for i in range(1,2):
    #     audio_input="./V0.2DataSet/wav/"+str(i)+"_video"
    #     diarization_nvidia_sortformer_process(audio_input,output_path)
    audio_input="./V0.2DataSet/wav/"+str(1)+"_video"
    diarization_nvidia_sortformer_process(audio_input,output_path)