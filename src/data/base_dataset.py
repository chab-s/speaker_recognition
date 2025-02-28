from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np


class BaseDataset(ABC):
    def __init__(self, data_path: str, **kwargs):
        self.data_path = data_path
        self.df = pd.DataFrame()
        self.df = None
        self.features = None
        self.labels = None
        self.encoder = LabelEncoder()
        self.kwargs = kwargs

    @abstractmethod
    def _load_audio_files(self):
        pass

    @abstractmethod
    def preprocess(self):
        pass

    def split_data(self, test_size: float, val_size: float, random_state: int = 42):

        X_train, y_train, X_test, y_test = train_test_split(
            self.features, self.labels,
            test_size=test_size,
            random_state=random_state,
            stratify=self.labels
        )
        X_train, y_train, X_val, y_val = train_test_split(
            X_train, y_train,
            test_size=(1-test_size)*val_size,
            random_state=random_state,
            stratify=y_train
        )

        return X_train, y_train, X_val, y_val, X_test, y_test

    def _encode_label(self, column_name: str = 'speaker'):
        self.labels = self.encoder.fit_transform(self.df[column_name])

    def decode_label(self, label_id: int):
        return self.encoder.inverse_transform(self.labels)[label_id]

    def get_batch(self, batch_size: int = 32, shuffle=True):
        pass
