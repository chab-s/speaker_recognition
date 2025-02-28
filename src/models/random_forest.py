from sklearn.ensemble import RandomForestClassifier
from .base_model import BaseModel
import joblib
import os
from sklearn.metrics import accuracy_score

class RandomForest(BaseModel):
    def __init__(self, dataset, n_estimators: int, test_size: float, val_size: float):
        super().__init__()
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = dataset.split_data(
            test_size=test_size, val_size=val_size)
        self.n_estimators = n_estimators
        self.model = self.build()

    def build(self):
        return RandomForestClassifier(n_estimators=self.n_estimators)

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Accuracy random_forest: {accuracy}")
        return accuracy

    def predict(self, X):
        y_pred = self.model.predict(X)
        return y_pred

    def save(self, path: str, filename: str = 'random_forest.pkl'):
        base, ext = os.path.splitext(filename)
        i = 1
        while os.path.exists(os.path.join(path, filename)):
            i += 1
            filename = f"{base}_{i}.{ext}"
        joblib.dump(self.model, os.path.join(path, filename))
        print('Model random_forest saved')

    def load(self, path: str, filename: str = 'random_forest.pkl'):
        self.model = joblib.load(os.path.join(path, filename))
        print("Model random_forest loaded")

