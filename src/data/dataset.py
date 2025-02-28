from .base_dataset import BaseDataset
from src.features.mfcc import mfcc

import os
import librosa
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
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
        self.labels = self._encode_labels(self.df, column_name='speaker')

    def _extract_features(self):
        self.df['mfccs'] = self.df.apply(lambda row: mfcc(row['signal'], row['sampling_rate']), axis=1)

class UbmGmmDataset(SpeakerDataset):
    def __init__(self, data_path: str, speakers):
        super().__init__(data_path, speakers)

    def split_data(self, gmm_size: float, test_size: float, random_state: int = 42):
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.labels,
            test_size=test_size,
            random_state=random_state,
            stratify=self.labels
        )

        X_ubm, X_gmm, y_ubm, y_gmm = train_test_split(
            self.features, self.labels,
            test_size=gmm_size,
            random_state=random_state,
            stratify=self.labels
        )

        return X_ubm, y_ubm,X_gmm, y_gmm, X_test, y_test

    def _encode_labels(self, df, column_name):
        labels = self.encoder.fit_transform(df[column_name])
        return labels

    def decoder_label(self, label):
        speakers = self.encoder.inverse_transform(label)
        return speakers


class SpeakerDataLoader(Dataset):
    def __init__(self, df):
        self.features = df.filter(like='mfcc').values.astype(np.float32)
        self.labels = df['speaker'].values.astype(np.int64)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]



