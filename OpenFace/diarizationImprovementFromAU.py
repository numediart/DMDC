import pandas as pd 
import os

def infer_face_id_to_speaker(AU_df, start_frame, end_frame):
    """
    Infer which face_id corresponds to which SPEAKER_XX by analyzing mouth activity (AU25, AU26)
    between start_frame and end_frame.
    """
    frames_scope = range(start_frame, end_frame + 1)
    score = {face_id: 0 for face_id in AU_df[" face_id"].unique()}
    for frame in frames_scope:
        frame_data = AU_df[AU_df["frame"] == frame]
        if len(frame_data) == 2:
            for _, row in frame_data.iterrows():
                face_id = int(row[" face_id"])
                au25 = row[' AU25_r']
                au26 = row[' AU26_r']
                score[face_id] += au25 + au26

    # face_id with higher AU activity is assumed to be the active speaker
    if score[0] > score[1]:
        return {0: "SPEAKER_00", 1: "SPEAKER_01"}
    else:
        return {1: "SPEAKER_00", 0: "SPEAKER_01"}



def diarizationImprovementFromAU(diarization_path,AU_path,output_path,_):
    diarization=pd.read_csv(diarization_path)
    AU=pd.read_csv(AU_path)
    # Removed unused variable
    max_frame=int(AU.iloc[-1]["frame"])

    for _,row in diarization.iterrows():
        print(row)
        # variables reset/def
        start_frame_row = row["start_frame"]
        end_frame_row = row["end_frame"]

        # Dynamically infer mapping between face_id and speaker for this segment
        face_to_speaker = infer_face_id_to_speaker(AU, start_frame_row, end_frame_row)

        lengh_segment=0
        AU25_r_speaker_0_mean=0
        AU25_r_speaker_1_mean=0
        AU26_r_speaker_0_mean=0
        AU26_r_speaker_1_mean=0
        AU25_r_speaker_0_above_threshold_count = 0
        AU25_r_speaker_1_above_threshold_count = 0

        # Frame scope
        frames_scope = range(start_frame_row, min(end_frame_row, max_frame + 1))
        print(frames_scope)

        for frame in frames_scope:
            current_frame_data = AU[AU["frame"] == frame]
            if current_frame_data.empty!= True:
                size=len(current_frame_data)
                # if dyadic
                if size==2:
                    # [["AU25_r", "AU26_r", "AU27_r"]]
                    # AU25 (Lips Part)
                    # AU26 (Jaw Drop)
                    speaker_0_data = current_frame_data[current_frame_data[" face_id"] == 0]
                    speaker_1_data = current_frame_data[current_frame_data[" face_id"] == 1]

                    if not speaker_0_data.empty and not speaker_1_data.empty:
                        lengh_segment += 1
                        AU25_0 = speaker_0_data[' AU25_r'].values[0]
                        AU25_1 = speaker_1_data[' AU25_r'].values[0]
                        AU26_0 = speaker_0_data[' AU26_r'].values[0]
                        AU26_1 = speaker_1_data[' AU26_r'].values[0]

                        AU25_r_speaker_0_mean += AU25_0
                        AU25_r_speaker_1_mean += AU25_1
                        AU26_r_speaker_0_mean += AU26_0
                        AU26_r_speaker_1_mean += AU26_1

                        if AU25_0 > 1.5:
                            AU25_r_speaker_0_above_threshold_count += 1
                        if AU25_1 > 1.5:
                            AU25_r_speaker_1_above_threshold_count += 1
        if lengh_segment != 0:
            AU25_r_speaker_0_mean /= lengh_segment
            AU25_r_speaker_1_mean /= lengh_segment
            AU26_r_speaker_0_mean /= lengh_segment
            AU26_r_speaker_1_mean /= lengh_segment


            # Presume speaker based on combined AU scores and thresholds
            sum_AU_speaker_0 = AU25_r_speaker_0_mean + AU26_r_speaker_0_mean
            sum_AU_speaker_1 = AU25_r_speaker_1_mean + AU26_r_speaker_1_mean


            AU25_r_speaker_0_above_threshold = AU25_r_speaker_0_above_threshold_count / lengh_segment * 100
            AU25_r_speaker_1_above_threshold = AU25_r_speaker_1_above_threshold_count / lengh_segment * 100

            
            # Consider both mean scores and percentage of frames above threshold
            score_speaker_0 = sum_AU_speaker_0 + AU25_r_speaker_0_above_threshold
            score_speaker_1 = sum_AU_speaker_1 + AU25_r_speaker_1_above_threshold

            # Get the actual face_id corresponding to higher score
            if score_speaker_0 > score_speaker_1:
                presumed_face_id = 0
            else:
                presumed_face_id = 1

            presumed_speaker = face_to_speaker[presumed_face_id]

            diarization.at[_, 'Presume_Speaker_From_AU'] = presumed_speaker
            diarization.at[_, 'Mean_AU25_r_Speaker_0'] = round(AU25_r_speaker_0_mean, 2)
            diarization.at[_, 'Mean_AU25_r_Speaker_1'] = round(AU25_r_speaker_1_mean, 2)
            diarization.at[_, 'Mean_AU26_r_Speaker_0'] = round(AU26_r_speaker_0_mean, 2)
            diarization.at[_, 'Mean_AU26_r_Speaker_1'] = round(AU26_r_speaker_1_mean, 2)
            diarization.at[_, '%Frames_AU25_r_Speaker_0_Above_1.5'] = round(AU25_r_speaker_0_above_threshold, 2)
            diarization.at[_, '%Frames_AU25_r_Speaker_1_Above_1.5'] = round(AU25_r_speaker_1_above_threshold, 2)

    # Save the updated diarization DataFrame to a CSV file
    updated_diarization_path = os.path.join(output_path, "updated_diarization.csv")
    diarization.to_csv(updated_diarization_path, index=False)
    # Select only the required columns and rename them
    updated_diarization_path = os.path.join(output_path, "pedicted_diarization.csv")
    diarization_subset = diarization[['start_frame', 'end_frame', 'Presume_Speaker_From_AU']].rename(
        columns={'Presume_Speaker_From_AU': 'speaker'}
    )
    diarization_subset.to_csv(updated_diarization_path, index=False)
    
    #                 sum_AU_speaker_0=(AU25_r_speaker_0+AU26_r_speaker_0)
    #                 sum_AU_speaker_1=(AU25_r_speaker_1+AU26_r_speaker_1)

    #                 if (max(sum_AU_speaker_0,sum_AU_speaker_1)>1):
    #                     if(sum_AU_speaker_0>sum_AU_speaker_1):
    #                         open_face_diar.append(
    #                             {
    #                                 "frame":int(current_frame_data.iloc[0]['frame']),
    #                                 "speaker":"SPEAKER_00"
    #                             }
    #                         )
    #                     else:
    #                         open_face_diar.append(
    #                             {
    #                                 "frame":int(current_frame_data.iloc[1]['frame']),
    #                                 "speaker":"_01"
    #                             }
    #                         )
    #                 else:
    #                     open_face_diar.append(
    #                             {
    #                                 "frame":int(current_frame_data.iloc[1]['frame']),
    #                                 "speaker":"none"
    #                             }
    #                         )
                    

    # # Create segments based on speaker changes
    # segments = []
    # if open_face_diar:
    #     current_speaker = open_face_diar[0]["speaker"]
    #     start_frame = open_face_diar[0]["frame"]

    #     for entry in open_face_diar[1:]:
    #         if entry["speaker"] != current_speaker:
    #             segment_duration = entry["frame"] - start_frame
    #             if segment_duration >= 3:  # Only add segments with duration >= 3 frames
    #                 segments.append({
    #                 "start_frame": start_frame,
    #                 "end_frame": entry["frame"] - 1,
    #                 "speaker": current_speaker
    #                 })
    #             current_speaker = entry["speaker"]
    #             start_frame = entry["frame"]

    #     # Add the last segment
    #     segments.append({
    #         "start_frame": start_frame,
    #         "end_frame": open_face_diar[-1]["frame"],
    #         "speaker": current_speaker
    #     })
        
    # # Save the segments list as a CSV file
    # segments_df = pd.DataFrame(segments)
    # segments_file_path = os.path.join(output_path, "segments_open_face_diar.csv")
    # segments_df.to_csv(segments_file_path, index=False)
    # # Save the open_face_diar list as a CSV file
    # output_df = pd.DataFrame(open_face_diar)
    # output_file_path = os.path.join(output_path, "open_face_diar.csv")
    # output_df.to_csv(output_file_path, index=False)




if __name__ == "__main__":

    
    diarization_path = os.path.join(os.path.dirname(__file__), "input", "6_video_diarization.csv")
    AU_path = os.path.join(os.path.dirname(__file__), "input", "6_video_AU.csv")
    output_path = os.path.join(os.path.dirname(__file__), "output")

    diarizationImprovementFromAU(diarization_path,AU_path,output_path,24)