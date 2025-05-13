import librosa
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os

def extractAndSaveMFCC(signal, folder, fileName):
        print(signal)
        mfcc = librosa.feature.mfcc(y=signal)
        librosa.display.specshow(mfcc)
        plt.savefig(folder + '/' + fileName + '.png')
        return mfcc
   

# Get the absolute path of the WAV file
wav_path = os.path.abspath('./V0DataSet/wav/1_video.wav')
signal, sr = librosa.load(wav_path)


extractAndSaveMFCC(signal, './V0DataSet/mffc', '1_video')
