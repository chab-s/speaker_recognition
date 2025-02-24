import os
import numpy as np
import pandas
from sklearn.model_selection import train_test_split

from src import SpeakerClassifier
from src import AudioDatasetLoader, SpeakerFeatureExtractor, NoiseFeatureExtractor
from src import ConfigManager
from src import SpeakerClustering
from src import Visualiser
from src import GMM_UBM
from sklearn.metrics import adjusted_rand_score, homogeneity_score, completeness_score


conf = ConfigManager('conf.json')

def train_model(save: bool = True):
    dataset_path = conf.get('database.database_path')
    speakers = conf.get('database.speakers')
    inference_data_path = conf.get('model.inference_data_path')

    model = conf.get('model.model_type')
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

    svm = SpeakerClassifier(model=model)
    svm.train(X_train, y_train)
    accuracy = svm.evaluate(X_test, y_test)
    print(f"Accuracy: {accuracy}")

    if save:
        svm.save_model(save_path)

def infer_model(weigths_path):
    dataset_path = conf.get('database.database_path')
    speakers = conf.get('database.speakers')
    inference_data_path = conf.get('model.inference_data_path')
    model = conf.get('model.model')

    dataset_speakers = SpeakerFeatureExtractor(dataset_path, speakers)
    dataset_speakers.features_extraction()

    X = np.load(os.path.join(inference_data_path, 'inference_data.npy'), allow_pickle=True)
    y_true = np.load(os.path.join(inference_data_path, 'inference_labels.npy'), allow_pickle=True)

    svm = SpeakerClassifier(model=model)
    svm.load_model(weigths_path)

    y_pred = svm.model.predict(X)
    accuracy = np.mean(y_pred == y_true)

    print(f"Prediction: {y_pred[:5]}, True label: {y_true[:5]}")
    print(f"Inference Accuracy: {accuracy:.4f}")

def use_clustering():
    dataset_path = conf.get('database.database_path')
    speakers = conf.get('database.speakers')
    inference_data_path = conf.get('model.inference_data_path')

    model = conf.get('model.model')
    test_size = conf.get('model.train_size')
    random_state = conf.get('model.random_state')
    save_path = conf.get('model.save_path')

    dataset_speakers = SpeakerFeatureExtractor(dataset_path, speakers)
    dataset_speakers.features_extraction()

    X = dataset_speakers.df.filter(like='mfcc').values
    y = dataset_speakers.df['speaker']

    clustering = SpeakerClustering(model=model, n_clusters=dataset_speakers.df["speaker"].nunique())
    clustering.test_scaler(np.stack(X))

    dataset_speakers.df['cluster'] = clustering.clusterization(X)

    evaluation = clustering.evaluate_clusters(dataset_speakers.df['speaker'], dataset_speakers.df['cluster'])

    # visualiser = Visualiser(dataset_speakers.df)
    # visualiser.mfcc_histogram()
    # visualiser.mfcc_boxplot()
    # visualiser.normality_test(test='anderson')

def ubm():
    dataset_path = conf.get('database.database_path')
    speakers = conf.get('database.speakers')
    n_components = conf.get('ubm_params.n_components')
    max_iter = conf.get('ubm_params.max_iter')
    random_state = conf.get('ubm_params.random_state')
    save_path = conf.get('ubm_params.save_path')
    inference_data_path = conf.get('ubm_params.inference_data_path')


    dataset_speakers = SpeakerFeatureExtractor(dataset_path, speakers)
    dataset_speakers.features_extraction()

    X = dataset_speakers.df.filter(like='mfcc').values
    y = dataset_speakers.df['speaker']

    df_train, df_test = train_test_split(
        dataset_speakers.df,
        test_size=0.2,
        random_state=42,
        stratify=dataset_speakers.df['speaker']
    )

    df_ubm, df_adapt = train_test_split(
        df_train,
        train_size=0.3,
        random_state=42,
        stratify=df_train['speaker']
    )
    print(f"UBM Training: {len(df_ubm)} samples")
    print(f"Speaker Adaptation: {len(df_adapt)} samples")
    print(f"Final Test: {len(df_test)} samples")

    X_inference = np.vstack(df_test.filter(like='mfcc').values)
    y_inference = np.vstack(df_test.filter(like='speaker').values)
    np.save(os.path.join(inference_data_path, 'ubm_inference_data.npy'), X_inference)
    np.save(os.path.join(inference_data_path, 'ubm_inference_labels.npy'), y_inference)

    gmm_ubm = GMM_UBM(n_components, max_iter, random_state)

    gmm_ubm.train_ubm(df_ubm)
    gmm_ubm.adapt_speaker_models(df_adapt)

    gmm_ubm.save_model(save_path)

def ubm_test():
    speakers = conf.get('database.speakers')
    n_components = conf.get('ubm_params.n_components')
    max_iter = conf.get('ubm_params.max_iter')
    random_state = conf.get('ubm_params.random_state')
    save_path = conf.get('ubm_params.save_path')
    inference_data_path = conf.get('ubm_params.inference_data_path')

    gmm_ubm = GMM_UBM(n_components, max_iter, random_state)
    gmm_ubm.load_model(save_path, speakers)

    X = np.load(os.path.join(inference_data_path, 'ubm_inference_data.npy'), allow_pickle=True)
    y = np.load(os.path.join(inference_data_path, 'ubm_inference_labels.npy'), allow_pickle=True)

    # best_speaker = gmm_ubm.predict_speaker(X[0])
    # print(f'true_speaker: {y[0]}, predict_speaker: {best_speaker}')

    score = 0
    for feature, speaker in zip(X,y):
        predict_speaker = gmm_ubm.predict_speaker(feature)
        if speaker == predict_speaker:
            score+=1
    print(f'Accuracy: {score/len(y)}')





if __name__ == '__main__':

    # train_model()
    # weights_path = conf.get("model.save_path")
    # infer_model(weights_path)

    # use_clustering()
    # ubm()
    ubm_test()



