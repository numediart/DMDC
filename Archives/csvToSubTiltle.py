import csv
import datetime
import os


def format_srt_time(seconds):
    td = datetime.timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    millis = int((td.total_seconds() - total_seconds) * 1000)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def csv_to_subtitle(csv_file, subtitle_file):
    os.makedirs(os.path.dirname(subtitle_file), exist_ok=True)
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header

        subtitle_lines = []
        index = 1
        for row in reader:
            start_time = float(row[0])
            end_time = float(row[1])
            speaker = row[2]

            start_srt = format_srt_time(start_time)
            end_srt = format_srt_time(end_time)

            subtitle_lines.append(f"{index}\n{start_srt} --> {end_srt}\n{speaker}\n\n")
            index += 1

    with open(subtitle_file, 'w', encoding='utf-8') as file:
        file.writelines(subtitle_lines)

# Utilisation
# csv_to_subtitle(
#     './V0DataSet/Diarization_Results/3_video.wav_diarization_results_20250512-150225.csv',
#     './V0DataSet/Subtitle/3_video.srt'
# )
