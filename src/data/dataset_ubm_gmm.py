from .dataset import SpeakerDataset
from sklearn.model_selection import train_test_split

class UbmGmmDataset(SpeakerDataset):
    def __init__(self, data_path: str, speakers):
        super().__init__(data_path, speakers)

    def split_data(self, gmm_size: float, test_size: float, random_state: int = 42):
        X_train, y_train, X_test, y_test = train_test_split(
            self.features, self.labels,
            test_size=test_size,
            random_state=random_state,
            stratify=self.labels
        )

        X_ubm, y_ubm, X_gmm, y_gmm = train_test_split(
            self.features, self.labels,
            test_size=gmm_size,
            random_state=random_state,
            stratify=self.labels
        )

        return X_ubm, y_ubm,X_gmm, y_gmm, X_test, y_test