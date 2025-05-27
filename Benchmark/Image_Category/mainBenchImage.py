import os
import cv2
import pandas as pd

inputBenchVideo = os.path.join(os.path.dirname(__file__), "../../V0DataSet/mp4/")
inputBenchSegments = os.path.join(os.path.dirname(__file__), "../../V0DataSet/mp4/")
outputBenchImage = os.path.join(os.path.dirname(__file__), "./output/")

def mainBenchImage():
    benchmarkTable=[
        { 
            "start_time":0,
            "end_time":1,
            "type":"Test",
            "valid":True,
    }
    ]
    for video in os.listdir(inputBenchVideo):
        print(video)
        start_sec = 5   
        end_sec = 10    
        video_type="dyadic"

        cap = cv2.VideoCapture(os.path.join(inputBenchVideo, video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_idx > end_frame:
                break
            if frame_idx >= start_frame:
                frame_resized = cv2.resize(frame, (600, 340))
                cv2.imshow('Video', frame_resized)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            frame_idx += 1
        cap.release()
        cv2.destroyAllWindows()

        user_input_for_validation = input("The videos was valid & only "+video_type+" or not (0: false, 1: true) : ")
        if user_input_for_validation=="1":
            user_input_for_validation=True
        else:
            user_input_for_validation=False
        print("You entered:", user_input_for_validation)
        benchmarkTable.append(
            { 
                "start_time":start_sec,
                "end_time":end_sec,
                "type":video_type,
                "valid":user_input_for_validation,
        }
        )

    df_bench=pd.DataFrame(benchmarkTable)
    print(df_bench)
    df_bench.to_csv(os.path.join(outputBenchImage, "benchmark_results.csv"))


mainBenchImage()