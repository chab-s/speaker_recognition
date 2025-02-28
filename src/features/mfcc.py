import numpy as np
import librosa

def mfcc(y, sr):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.hstack((np.mean(mfccs, axis=1), np.std(mfccs, axis=1)))