import pandas as pd
import os 

def typeOfSceneDetectionAU(AU_output,output_path=None):
    AU=pd.read_csv(AU_output)
    max_frame=int(AU.iloc[-1]["frame"])
    type_of_scene=[]
    for frame in range(max_frame):
        current_frame_data = AU[AU["frame"] == frame]
        if current_frame_data.empty!= True:#there is something
            size=len(current_frame_data)
            # dyadic
            if size==2:
                type_of_scene.append(
                    {
                        "frame":frame,
                        "type":"dyadic"
                    }
                )
            # single
            elif size==1:
                type_of_scene.append(
                    {
                        "frame":frame,
                        "type":"single"
                    }
                )
            # other
            else:
                type_of_scene.append(
                    {
                        "frame":frame,
                        "type":"other"
                    }
                )
        else:
            type_of_scene.append(
                {
                    "frame":frame,
                    "type":"other"
                }
            )
    
    output_df = pd.DataFrame(type_of_scene)
    if output_path!=None:
        output_file = os.path.join(output_path, "scene_types.csv")
        output_df.to_csv(output_file, index=False)
    df_type_of_scene=output_df
    return df_type_of_scene
    
def typeOfSceneFrameToSegments(df_type_of_scene,output_path,fps):
    segments = []
    current_segment = {"start_frame": None, "end_frame": None,"start_sec": None, "end_sec": None, "type": None}

    for index, row in df_type_of_scene.iterrows():
        frame_type = row["type"]
        frame = row["frame"]

        # New segments
        if current_segment["type"] is None:
            start_frame=current_segment["start_frame"] = frame
            current_segment["start_sec"] = round(start_frame/30,2)
            current_segment["type"] = frame_type
        # Change detected
        elif current_segment["type"] != frame_type:
            end_frame=current_segment["end_frame"] = frame - 1
            current_segment["end_sec"] = round(end_frame/30,2)
            if(current_segment["end_sec"]-current_segment["start_sec"]>1):
                segments.append(current_segment)
            current_segment = {"start_frame": frame, "end_frame": None,"start_sec": None, "end_sec": None, "type": None}

    if current_segment["type"] is not None:
        current_segment["end_frame"] = df_type_of_scene.iloc[-1]["frame"]
        current_segment["end_sec"] = round(int(df_type_of_scene.iloc[-1]["frame"])/fps,2)

        segments.append(current_segment)

    output_df = pd.DataFrame(segments)
    output_file = os.path.join(output_path, "scene_types_segments.csv")
    output_df.to_csv(output_file, index=False)

    return output_df


if __name__ == "__main__":
    
    input_path = os.path.join(os.path.dirname(__file__), "input", "test.csv")
    output_path = os.path.join(os.path.dirname(__file__), "output")

    df_type_of_scene=typeOfSceneDetectionAU(input_path)
    typeOfSceneFrameToSegments(df_type_of_scene,output_path,24)