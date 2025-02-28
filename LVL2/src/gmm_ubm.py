from .Visualiser import Visualiser
import numpy as np
from sklearn.mixture import GaussianMixture
import joblib
import os

import torch.nn as nn

from .config_manager import ConfigManager
conf = ConfigManager('conf.json')


class GMM_UBM:
    def __init__(self, n_components: int = 64, max_iter: int = 100, random_stats: int = 42):
        self.n_components = conf.get('ubm_params.n_components'),
        self.max_iter = conf.get('ubm_params.max_iter'),
        self.covariance_type = conf.get('ubm_params.covariance_type'),
        self.random_state = conf.get('ubm_params.random_state'),
        self.ubm = None
        self.speaker_models = {}


    def train_ubm(self, df):
        X = np.vstack(df.filter(like='mfcc').values)
        self.ubm = GaussianMixture(
            n_components=self.n_components[0],
            max_iter=self.max_iter[0],
            covariance_type=self.covariance_type[0],
            random_state=self.random_state[0]
        )
        print(self.max_iter)
        print(self.covariance_type)
        self.ubm.fit(X)
        # Metrics
        log_likelihood = self.ubm.score(X)
        bic = self.ubm.bic(X)
        aic = self.ubm.aic(X)
        print("UBM trained successfully")
        print(f"Log-Vraisemblance Moyenne : {log_likelihood:.2f}")
        print(f"BIC : {bic:.2f}  /  AIC : {aic:.2f}")

        return log_likelihood, bic, aic

    def adapt_speaker_models(self, df):
        speakers = df['speaker'].unique()
        speaker_models = {}

        for speaker in speakers:
            df_speaker = df[df['speaker'] == speaker]
            X_speaker = np.vstack(df_speaker.filter(like='mfcc').values)

            gmm_speaker = GaussianMixture(
                n_components=self.n_components[0],
                max_iter=self.max_iter[0],
                covariance_type=self.covariance_type[0],
                random_state=self.random_state[0]
            )

            gmm_speaker.means_ = self.ubm.means_
            gmm_speaker.covariances_ = self.ubm.covariances_
            gmm_speaker.weights_ = self.ubm.weights_

            gmm_speaker.fit(X_speaker)
            speaker_models[speaker] = gmm_speaker

        self.speaker_models = speaker_models
        print('Adaptation of speaker GMMs complete')

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



class DNN_UBM(nn.Module):
    def __init__(self, input_dim, n_components):
        super(DNN_UBM, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, n_components)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return self.softmax(x)
