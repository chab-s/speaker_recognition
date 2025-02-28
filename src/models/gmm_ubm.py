from .base_model import BaseModel
from sklearn.mixture import GaussianMixture
import joblib
import os
import numpy as np

class GMM_UBM(BaseModel):
    def __init__(self, dataset, gmm_size: int, test_size: int, n_components: int, max_iter: int, covariance_type: str, random_state: int):
        super().__init__()
        self.dataset = dataset
        self.X_ubm, self.y_ubm, self.X_gmm, self.y_gmm, self.X_test, self.y_test = dataset.split_data(test_size=test_size, gmm_size=gmm_size)
        self.n_components= n_components
        self.max_iter = max_iter
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.ubm = None
        self.speaker_models = {}

    def build(self):
        self.ubm = GaussianMixture(
            n_components=self.n_components,
            max_iter=self.max_iter,
            covariance_type=self.covariance_type,
            random_state=self.random_state
        )

    def build_gmm(self):
        gmm_speaker = GaussianMixture(
            n_components=self.n_components,
            max_iter=self.max_iter,
            covariance_type=self.covariance_type,
            random_state=self.random_state
        )
        return gmm_speaker

    def train(self):
        self.ubm.fit(self.X_ubm)
        print(f"Model UBM trained successfully")
        for speaker in self.dataset.encoder.transform(self.dataset.speakers):
            X_speaker = np.array([X for X, y in zip(self.X_gmm, self.y_gmm) if speaker == y])
            gmm_speaker = self.build_gmm()
            gmm_speaker.means_ = self.ubm.means_
            gmm_speaker.covariances_ = self.ubm.covariances_
            gmm_speaker.weights_ = self.ubm.weights_

            gmm_speaker.fit(X_speaker)
            self.speaker_models[speaker] = gmm_speaker
            print(f"Model gmm for {speaker} trained successfully")

    def evaluate(self):
        accuracy = 0
        for features, true_speaker in zip(self.X_test, self.y_test):
            features = features.reshape(1, -1)
            log_likelihoods = {speaker: model.score(features) for speaker, model in self.speaker_models.items()}
            pred_speaker = max(log_likelihoods, key=log_likelihoods.get)
            if pred_speaker == true_speaker:
                accuracy += 1

        res = accuracy/self.X_test.shape[0]
        print(f"Accuracy: {res}")
        return res

    def predict(self, X):
        log_likelihoods = {speaker: model.score(X) for speaker, model in self.speaker_models.items()}
        pred_speaker = max(log_likelihoods, key=log_likelihoods.get)
        return pred_speaker

    def save(self, path: str, filename: str = 'ubm_gmm.pkl'):
        base, ext = os.path.splitext(filename)
        i = 0
        while os.path.exists(os.path.join(path, filename)):
            i += 1
            filename = f"{base}_{i}.{ext}"
        joblib.dump(self.ubm, os.path.join(path, filename))
        print('Ubm model saved')
        for speaker, model in self.speaker_models.items():
            joblib.dump(model, os.path.join(path, f'{speaker}_{i}.pkl'))
        print('Gmm speakers models saved')

    def load(self, path, filename: str = 'ubm_0.pkl'):
        self.ubm = joblib.load(filename)
        i = filename.split("_")[1].split(".")[0]
        self.speaker_models = {speaker: joblib.load(os.path.join(path, f"{speaker}_{i}.pkl")) for speaker in self.dataset.speakers}




