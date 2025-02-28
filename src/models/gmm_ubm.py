from .base_model import BaseModel
from sklearn.mixture import GaussianMixture
import joblib
import os

class GMM_UBM(BaseModel):
    def __init__(self, dataset, gmm_size: int, test_size: int, n_components: int, max_iter: int, covariance_type: str, random_state: int):
        super().__init__()
        self.dataset = dataset
        self.X_ubm, self.y_ubm, self.X_gmm, self.y_gmm, self.X_test, self.y_test = dataset.split(test_size=test_size, gmm_size=gmm_size)
        self.n_components= n_components
        self.max_iter = max_iter
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.ubm = None
        self.gmm_speaker = None
        self.speaker_models = {}

    def build(self):
        self.ubm = GaussianMixture(
            n_components=self.n_components,
            max_iter=self.max_iter,
            covariance_type=self.covariance_type,
            random_state=self.random_state
        )

        self.gmm_speaker = GaussianMixture(
            n_components=self.n_components,
            max_iter=self.max_iter,
            covariance_type=self.covariance_type,
            random_state=self.random_state
        )

    def train(self):
        self.ubm.fit(self.X_ubm)

        for speaker in self.dataset.speakers:
            gmm_speaker = self.gmm_speaker.copy()
            gmm_speaker.means_ = self.ubm.means_
            gmm_speaker.covariances_ = self.ubm.covariances_
            gmm_speaker.weights_ = self.ubm.weights_

            gmm_speaker.fit(self.X_gmm)
            self.speaker_models[speaker] = self.gmm_speaker

    def evaluate(self):
        accuracy = 0
        for features, true_speaker in zip(self.X_test, self.y_test):
            log_likelihoods = {speaker: model.score(features) for speaker, model in self.speaker_models.items()}
            pred_speaker = max(log_likelihoods, key=log_likelihoods.get)
            if pred_speaker == true_speaker:
                accuracy += 1

        res = accuracy/self.X_test.shape
        print(f"Accuracy: {res}")
        return res

    def save(self, path: str, filename: str = 'ubm_0.pkl'):
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




