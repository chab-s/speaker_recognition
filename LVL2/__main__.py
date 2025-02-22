import os
import numpy as np
from sklearn.model_selection import train_test_split

from src import SpeakerClassifier
from src import AudioDatasetLoader, SpeakerFeatureExtractor, NoiseFeatureExtractor
from src import ConfigManager

conf = ConfigManager('conf.json')

def train_model(save: bool = True):
    dataset_path = conf.get('database.database_path')
    speakers = conf.get('database.speakers')
    inference_data_path = conf.get('model.inference_data_path')

    model_type = conf.get('model.model_type')
    test_size = conf.get('model.train_size')
    random_state = conf.get('model.random_state')
    save_path = conf.get('model.save_path')

    dataset_speakers = SpeakerFeatureExtractor(dataset_path, speakers)
    dataset_speakers.features_extraction()

    X = dataset_speakers.df.filter(like='mfcc').values
    y = dataset_speakers.df['speaker']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    X_train, X_inference, y_train, y_inference = train_test_split(X_train, y_train, test_size=0.1, random_state=random_state)

    np.save(os.path.join(inference_data_path, 'inference_data.npy'), X_inference)
    np.save(os.path.join(inference_data_path, 'inference_labels.npy'), y_inference)

    svm = SpeakerClassifier(model_type=model_type)
    svm.train(X_train, y_train)
    accuracy = svm.evaluate(X_test, y_test)
    print(f"Accuracy: {accuracy}")

    if save:
        svm.save_model(save_path)

def infer_model(weigths_path):
    dataset_path = conf.get('database.database_path')
    speakers = conf.get('database.speakers')
    inference_data_path = conf.get('model.inference_data_path')
    model_type = conf.get('model.model_type')

    dataset_speakers = SpeakerFeatureExtractor(dataset_path, speakers)
    dataset_speakers.features_extraction()

    X = np.load(os.path.join(inference_data_path, 'inference_data.npy'), allow_pickle=True)
    y_true = np.load(os.path.join(inference_data_path, 'inference_labels.npy'), allow_pickle=True)

    svm = SpeakerClassifier(model_type=model_type)
    svm.load_model(weigths_path)

    y_pred = svm.model.predict(X)
    accuracy = np.mean(y_pred == y_true)

    print(f"Prediction: {y_pred[:5]}, True label: {y_true[:5]}")
    print(f"Inference Accuracy: {accuracy:.4f}")

if __name__ == '__main__':

    # train_model()
    weights_path = conf.get("model.save_path")
    infer_model(weights_path)






