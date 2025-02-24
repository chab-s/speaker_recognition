import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler
from collections import defaultdict
import joblib
import os

class GMM_UBM:
    def __init__(self, n_components: int = 64, max_iter: int = 100, random_stats: int = 42):
        self.scaler = RobustScaler()
        self.n_components = n_components
        self.max_iter = max_iter
        self.random_state = random_stats
        self.ubm = None
        self.speaker_models = {}

    def train_ubm(self, df):
        X = np.vstack(df.filter(like='mfcc').values)

        X_narmalized = self.scaler.fit_transform(X)

        self.ubm = GaussianMixture(
            n_components=self.n_components,
            max_iter=self.max_iter,
            covariance_type='diag',
            random_state=self.random_state
        )

        self.ubm.fit(X_narmalized)
        print("UBM trained successfully")

        return self.ubm

    def adapt_speaker_models(self, df):
        speakers = df['speaker'].unique()
        speaker_models = {}

        for speaker in speakers:
            X_speaker = np.vstack(df.filter(like='mfcc').values)
            X_normalized = self.scaler.transform(X_speaker)

            gmm_speaker = GaussianMixture(n_components=self.n_components,
            max_iter=self.max_iter,
            covariance_type='diag',
            random_state=self.random_state
            )

            gmm_speaker.means_ = self.ubm.means_
            gmm_speaker.covariances_ = self.ubm.covariances_
            gmm_speaker.weights_ = self.ubm.weights_

            gmm_speaker.fit(X_normalized)
            speaker_models[speaker] = gmm_speaker

        self.speaker_models = speaker_models
        print('Adaptation of speaker GMMs complete')

        return speaker_models

    def predict_speaker(self, mfccs):
        mfccs = mfccs.reshape(1, -1)
        log_likelihoods = {speaker: model.score(mfccs) for speaker, model in self.speaker_models.items()}
        return max(log_likelihoods, key=log_likelihoods.get)

    def save_model(self, save_path):
        joblib.dump(self.ubm, os.path.join(save_path, 'ubm.pkl'))
        for speaker, model in self.speaker_models.items():
            joblib.dump(model, os.path.join(save_path, f'{speaker}.pkl'))
        print('Model saved')

    def load_model(self, save_path, speakers):
        self.ubm = joblib.load(os.path.join(save_path, 'ubm.pkl'))
        self.speaker_models = {speaker: joblib.load(os.path.join(save_path, f'{speaker}.pkl')) for speaker in speakers}
        print('Model loaded')

