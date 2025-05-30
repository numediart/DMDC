import os
import cv2
import pandas as pd
from datetime import datetime
from pydub import AudioSegment
from pydub.playback import play
import threading


# Benchmark Parameter and path
inputBenchAudio = os.path.join(os.path.dirname(__file__), "../../V0DataSet/wav/")
inputBenchVideo = os.path.join(os.path.dirname(__file__), "../../V0DataSet/mp4/")
inputBenchSegments = os.path.join(os.path.dirname(__file__), "../../V0DataSet/segments/")
outputBenchAudio = os.path.join(os.path.dirname(__file__), "./output/")

def play_segment(cap, start_frame, end_frame, fps, audio_file, start_sec, end_sec, play_speed=2):
    # Start audio playback in a separate thread
    audio_thread = threading.Thread(target=playAudioTimeStamp, args=(audio_file, start_sec*1000, end_sec*1000))
    audio_thread.start()
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx > end_frame:
            break
        if frame_idx >= start_frame:
            # Resize to 480p (854x480)
            frame_480p = cv2.resize(frame, (854, 480))
            cv2.imshow('Video', frame_480p)
            # Play at play_speed: reduce waitKey delay (original: 1000/fps ms)
            delay = max(1, int((1000 / fps) / play_speed))
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break
        frame_idx += 1
    cap.release()
    cv2.destroyAllWindows()
    
    # Wait for audio to finish
    audio_thread.join()



def playAudioTimeStamp(audiofile,start_ms,end_ms):
    sound = AudioSegment.from_file(audiofile, format="wav")
    splice = sound[start_ms:end_ms]
    play(splice)




def mainBenchDiarization(play_speed=2):

    benchmarkTable=[]
    indexList=0
    print("Please Choose a video to benchmark:")
    for audio in os.listdir(inputBenchAudio):
        print(indexList," : ",audio)
        indexList+=1

    idSelected=int(input("Which video have u choose :"))
    audio=os.listdir(inputBenchAudio)[idSelected]


    id=audio.split("_")[0]
    segments_file=pd.read_csv(os.path.join(inputBenchSegments,str(id)+"_segments.csv"))
    # Cut segments in smaller filer
    # segments_file = generate_segments(video, segments_file)
    print(segments_file)
    segments_file_size = len(segments_file)
    for _, segment in segments_file.iterrows():
        # print(video)
        start_sec = segment['start_time']
        end_sec = segment['end_time']
        speaker = segment['speaker']

        
        print(
            "\n"*20,
            "="*60,
            "\n",
            "\t\tBenchmark Diarization]",
            "\n",
            "\t ℹ️  Audio N°", str(id),"Audio Name :", audio,"segments n°:",_,"/",str(segments_file_size),"\n ",
            "\t ⏳ Start sec=", round(start_sec,1),"\n",
            "\t ⌛ End sec=", round(end_sec,1),"\n",
            "\t 📼 Speaker: ", speaker,
            "\n",
            "="*60
        )


        playAudioTimeStamp(os.path.join(inputBenchAudio, audio),start_sec*1000,end_sec*1000)

        # cap = cv2.VideoCapture(os.path.join(inputBenchVideo, id+"_video.mp4"))
        # fps = cap.get(cv2.CAP_PROP_FPS)
        # start_frame = int(start_sec * fps)
        # end_frame = int(end_sec * fps)
        # play_segment(cap, start_frame, end_frame, fps,os.path.join(inputBenchAudio, audio) ,start_sec,end_sec,1.1)



        # USER INPUT "while"  0️⃣: false, 1️⃣: true, 2️⃣: replay, 3️⃣: save)
        while True:
            user_input_for_validation = input(
                "The video was valid & only " + str(speaker) +
                " or not (0️⃣: false, 1️⃣: true, 2️⃣: replay, 3️⃣: save) : "
            )
            if user_input_for_validation == "1":
                user_input_for_validation = True
                break
            elif user_input_for_validation == "2":
                # Replay the segment
                playAudioTimeStamp(os.path.join(inputBenchAudio, audio),start_sec*1000,end_sec*1000)
            elif user_input_for_validation == "3":
                # Save benchmark table so far
                df_bench = pd.DataFrame(benchmarkTable)
                now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                df_bench.to_csv(os.path.join(outputBenchAudio, "benchmark_results_" + now + ".csv"))
                print("Benchmark table saved.")
            else:
                user_input_for_validation = False
                break

        print("You entered:", user_input_for_validation)

        benchmarkTable.append(
        {
            "audio": audio,
            "start_time": start_sec,
            "end_time": end_sec,
            "speaker": speaker,
            "valid": user_input_for_validation,
        }
        )


    user_input_for_validation = input("The videos was valid & only from"+speaker+" or not (0: false, 1: true) : ")
    if user_input_for_validation=="1":
        user_input_for_validation=True
    else:
        user_input_for_validation=False

    print("You entered:", user_input_for_validation)
    benchmarkTable.append(
        { 
            "audio": audio,
            "start_time": start_sec,
            "end_time": end_sec,
            "speaker": speaker,
            "valid": user_input_for_validation,
    }
    )


    # Benchmark export
    df_bench=pd.DataFrame(benchmarkTable)
    print(df_bench)
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    df_bench.to_csv(os.path.join(outputBenchAudio, "benchmark_results_" + now + ".csv"))


mainBenchDiarization()