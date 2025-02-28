from sklearn.svm import SVC
from .base_model import BaseModel
import joblib
import os
from sklearn.metrics import accuracy_score

class SVMClassifier(BaseModel):
    def __init__(self, dataset, kernel: str, test_size: float, val_size: float):
        super().__init__()
        self.kernel = kernel
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = dataset.split_data(
            test_size=test_size, val_size=val_size)

        self.model = self.build()

    def build(self):
        return SVC(kernel=self.kernel)

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        y_pred = self.model.predict(self.X_test)
        return accuracy_score(self.y_test, y_pred)

    def save_model(self, path, filename: str = 'svm_0.pkl'):
        base, ext = os.path.splitext(filename)
        i = 0
        while os.path.exists(filename):
            i+=1
            filename = f"{base}_{i}.{ext}"
        joblib.dump(self.model, os.path.join(path, filename))
        print('Model saved')

    def load_model(self, save_path, filename: str = 'svm_0.pkl'):
        self.model = joblib.load(os.path.join(save_path, filename))
        print('Model loaded')