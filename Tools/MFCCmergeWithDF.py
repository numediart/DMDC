import pandas as pd
import os

def MFCCmergeWithDF(segments_csv,mfcc,outputPath,framePerSecond=30):
    segments_csv["MFCC"]=None
    for segment_start,segment_end,index_segment in zip(segments_csv["start_time"],segments_csv["end_time"],range(len(segments_csv["end_time"]))):
        # print(int(segment_start),int(segment_end),index_segment)
        MFCCFramePerFrame=[]
        for index_frames in range(int(segment_start*framePerSecond),int(segment_end*framePerSecond)):
            MFCCFramePerFrame.append(mfcc[str(index_frames)].to_list())
        # print(MFCCFramePerFrame)
        segments_csv.at[index_segment, "MFCC"]=MFCCFramePerFrame

    segments_csv.to_csv(outputPath)
    return segments_csv



#DEBUG CODE
# segments_csv = pd.read_csv("./V0DataSet/segments/1_segments.csv")
# segments_csv["Unnamed"] = None
# segments_csv.to_csv("./V0DataSet/segments/1_segments.csv", index=False)
    
# segments_csv= pd.read_csv("./V0DataSet/segments/1_segments.csv")
# mfcc= pd.read_csv("./V0DataSet/mfcc/1_video_mfcc.csv")
# mfccExportPath=os.path.join(os.path.dirname(__file__),"V0DataSet/segments/","1_segments.csv")
# MFCCmergeWithDF(segments_csv,mfcc,mfccExportPath)
    