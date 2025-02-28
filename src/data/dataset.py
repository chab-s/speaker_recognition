from .base_dataset import BaseDataset
from src.features.mfcc import mfcc

import os
import librosa
import numpy as np
import pandas as pd

from torch.utils.data import Dataset


class SpeakerDataset(BaseDataset):
    def __init__(self, data_path: str, speakers):
        super().__init__(data_path)
        self.speakers = speakers
        self.df = self._load_audio_files()
        self.shuffle = self.kwargs.get('shuffle', False)
        self.preprocess()

    def _load_audio_files(self):
        data = []
        for dir in os.listdir(self.data_path):
            if dir in self.speakers:
                audio_path = os.path.join(self.data_path, dir)
                if os.path.isdir(audio_path):
                    audio_files = [f for f in os.listdir(audio_path) if f.endswith(('.wav', '.mp3'))]
                    for audio in audio_files:
                        file_path = os.path.join(audio_path, audio)
                        y, sr = librosa.load(file_path, sr=None)
                        duration = librosa.get_duration(y=y, sr=sr)

                        data.append(
                            {
                                "speaker": dir,
                                "filename": audio,
                                "path": audio_path,
                                "signal": y,
                                "sampling_rate": sr,
                                "duration": duration
                            }
                        )

        return pd.DataFrame(data)

    def preprocess(self):
        self._extract_features()
        if self.shuffle:
            self.df.sample(frac=1, random_state=42, ignore_index=True, inplace=True)
        self.features = np.vstack(self.df['mfccs'].values)
        self._encode_label(column_name='speaker')

    def _extract_features(self):
        self.df['mfccs'] = self.df.apply(lambda row: mfcc(row['signal'], row['sampling_rate']), axis=1)

class SpeakerDataLoader(Dataset):
    def __init__(self, df):
        self.features = df.filter(like='mfcc').values.astype(np.float32)
        self.labels = df['speaker'].values.astype(np.int64)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]



