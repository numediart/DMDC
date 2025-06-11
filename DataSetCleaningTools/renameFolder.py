import os


for i in range(1,10):
    listener_dir = './64_frames_windowed_clips/'+str(i)+'_video/speaker/'

    for filename in os.listdir(listener_dir):
        if filename.endswith('.csv'):
            # Example: 87.65_to_193.76_segment_listener_1_64_0_64frames_csv.csv
            try:
                start_sec = float(filename.split("_")[0])  # start frame
                start_frm = int(start_sec*30)+int(filename.split("_")[5])  # start frame
                end_frm =  int(start_sec*30)+ int(filename.split("_")[6])    # end frame


                new_filename = f"speaker_AU_{start_frm}_to_{end_frm}.csv"
                old_path = os.path.join(listener_dir, filename)
                new_path = os.path.join(listener_dir, new_filename)
                os.rename(old_path, new_path)
                print(f"\nRenamed {filename} -> {new_filename}")
            except Exception as e:
                print(f"Skipping {filename}: {e}")