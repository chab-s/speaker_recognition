import os
import numpy as np
import pandas as pd
import librosa
import json
from .config_manager import ConfigManager
conf = ConfigManager("conf.json")

from torch.utils.data import Dataset


class AudioDatasetLoader:

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.df = self._load_audio_files()

    def _load_audio_files(self):
        data = []
        for dir in os.listdir(self.dataset_path):
            audio_path = os.path.join(self.dataset_path, dir)
            if os.path.isdir(audio_path):
                audio_files = [f for f in os.listdir(audio_path) if f.endswith(('.wav', '.mp3'))]
                for audio in audio_files:
                    file_path = os.path.join(audio_path, audio)
                    y, sr = librosa.load(file_path, sr=None)
                    duration = librosa.get_duration(y=y, sr=sr)

                    data.append(
                        {
                            "AudioCorpus": dir,
                            "filename": audio,
                            "path": audio_path,
                            "signal": y,
                            "sampling_rate": sr,
                            "duration": duration
                         }
                    )

        return pd.DataFrame(data)

class SpeakerFeatureExtractor(AudioDatasetLoader):
    def __init__(self, dataset_path, speakers):
        super().__init__(dataset_path)
        self.df = self.df[self.df['AudioCorpus'].isin(speakers)].copy()
        self.df = self.df.rename(columns={'AudioCorpus': 'speaker'})

    def adapte_duration(self, target_duration: int):
        data, new_signals, new_durations, files, paths = [], [], [], [], []
        for speaker, group in self.df.groupby('speaker'):
            sampling_rate = self.df['sampling_rate'].iloc[0]
            target_sample = int(target_duration * sampling_rate)
            temp_signal = np.array([])

            for _, row in group.iterrows():
                temp_signal = np.concatenate([temp_signal, row['signal']])
                files.append(self.df['filename'])
                paths.append(self.df['path'])
                if len(temp_signal) >= target_sample:
                    new_signals = temp_signal[:target_sample]
                    new_durations = len(new_signals) / sampling_rate
                    data.append(
                        {
                            "speaker": speaker,
                            "filename": files,
                            "path": paths,
                            "signal": new_signals,
                            "sampling_rate": target_sample,
                            "duration": new_durations
                        }
                    )

                    temp_signal = new_signals[target_sample:]

        return pd.DataFrame(data)

    def features_extraction(self):
        def mfcc_mean(y, sr):
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=conf.get('database.n_mfcc'))
            mfccs = np.hstack((np.mean(mfccs, axis=1), np.std(mfccs, axis=1)))
            return mfccs

        mfcc_features = np.vstack(self.df.apply(lambda row: mfcc_mean(row['signal'], row['sampling_rate']), axis=1))

        num_coeffs = mfcc_features.shape[1]
        mfcc_columns = [f'mfcc_{i}' for i in range(num_coeffs)]

        mfcc_df = pd.DataFrame(mfcc_features, columns=mfcc_columns)
        self.df = pd.concat([self.df.reset_index(drop=True), mfcc_df], axis=1)

class NoiseFeatureExtractor(AudioDatasetLoader):
    """Classe pour extraire les caractéristiques des bruits audio."""

    def __init__(self, dataset_path, noises):
        super().__init__(dataset_path)
        self.df = self.df[self.df['AudioCorpus'].isin(noises)].copy()


class SpeakerDataLoader(Dataset):
    def __init__(self, df):
        self.features = df.filter(like='mfcc').values.astype(np.float32)
        self.labels = df['speaker'].values.astype(np.int64)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]



