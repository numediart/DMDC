import librosa
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os

def extractAndSaveMFCC(signal, folder, fileName):
    """
    Extracts Mel-frequency cepstral coefficients (MFCC) from a given audio signal and saves the corresponding spectrogram plot as a PNG file.
    Parameters:
    - signal: numpy.ndarray
        The input audio signal.
    - folder: str
        The folder path where the PNG file will be saved.
    - fileName: str
        The name of the PNG file (without extension).
    Returns:
    - mfcc: numpy.ndarray
        The extracted MFCC features.
    """
    
    mfcc = librosa.feature.mfcc(y=signal)
    librosa.display.specshow(mfcc)
    plt.savefig(folder + '/' + fileName + '.png')
    return mfcc
   


# # Example usage
# wav_path = os.path.abspath('./V0DataSet/wav/1_video.wav')
# signal, sr = librosa.load(wav_path)


# extractAndSaveMFCC(signal, './V0DataSet/mffc', '1_video')
