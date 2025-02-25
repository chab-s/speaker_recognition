from .Visualiser import Visualiser
import numpy as np
from sklearn.mixture import GaussianMixture
import joblib
import os

from .config_manager import ConfigManager
conf = ConfigManager('conf.json')


class GMM_UBM:
    def __init__(self, n_components: int = 64, max_iter: int = 100, random_stats: int = 42):
        self.n_components = n_components
        self.max_iter = max_iter
        self.random_state = random_stats
        self.ubm = None
        self.speaker_models = {}
        self.vis = Visualiser()

    def train_ubm(self, df):
        X = np.vstack(df.filter(like='mfcc').values)
        self.ubm = GaussianMixture(
            n_components=self.n_components,
            max_iter=self.max_iter,
            covariance_type=conf.get('gmm_ubm.covariance_type'),
            random_state=self.random_state
        )

        self.ubm.fit(X)
        # Metrics
        log_likelihood = self.ubm.score(X)
        bic = self.ubm.bic(X)
        aic = self.ubm.aic(X)
        print("UBM trained successfully")
        print(f"Log-Vraisemblance Moyenne : {log_likelihood:.2f}")
        print(f"BIC : {bic:.2f}  /  AIC : {aic:.2f}")
        # Visualisation
        return log_likelihood, bic, aic

    def adapt_speaker_models(self, df):
        speakers = df['speaker'].unique()
        speaker_models = {}

        for speaker in speakers:
            df_speaker = df[df['speaker'] == speaker]
            X_speaker = np.vstack(df_speaker.filter(like='mfcc').values)

            gmm_speaker = GaussianMixture(n_components=self.n_components,
            max_iter=self.max_iter,
            covariance_type=conf.get('gmm_ubm.covariance_type'),
            random_state=self.random_state
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



class DNN_UBM():
    def __init__(self, data):
        self.model = None
        self.data = data


# def main(self):
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(self.model.parameters(), lr=conf.get('dnn_ubm.lr'))
#
#     self.model.train()
#     for epoch in range(conf.get('dnn_ubm.num_epochs')):
#         epoch_loss = 0.0
#         for batch_features, batch_labels in self.data:
#             batch_features = torch.tensor(batch_features)
#             batch_labels = torch.tensor(batch_labels)
#
#             optimizer.zero_grad()
#             output = self.model(batch_features)
#             loss = criterion(output, batch_labels)
#             loss.backward()
#             optimizer.step()
#
#             epoch_loss += loss.item() * batch_features.size(0)
#         avg_loss = epoch_loss / len(self.data.dataset)
