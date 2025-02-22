from sklearn.svm import SVC
import pickle
from sklearn.metrics import accuracy_score

class SpeakerClassifier:
    def __init__(self, model_type: str):
        if model_type == 'SVM':
            self.model = SVC(kernel="linear")

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def evaluate(self, x_test, y_test):
        y_pred = self.model.predict(x_test)
        return accuracy_score(y_test, y_pred)

    def save_model(self, filename):
        with open(filename, 'wb') as file:
             pickle.dump(self.model, file)

    def load_model(self, filename):
        with open(filename, 'rb') as file:
            self.model = pickle.load(file)
