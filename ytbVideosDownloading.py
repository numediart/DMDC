from pytubefix import YouTube


yt = YouTube('https://www.youtube.com/watch?v=xb3HFK2Gngk')

yt.streams.filter().get_by_resolution("360p").download(output_path="./V0DataSet/mp4", filename='4_video.mp4')
yt.streams.filter().get_audio_only().download(output_path="./V0DataSet/wav", filename='4_video.wav')