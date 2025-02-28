from src import SpeakerClassifier
from src import SpeakerClustering

from src import SpeakerFeatureExtractor, SpeakerDataLoader
from src import ConfigManager
from src import GMM_UBM, DNN_UBM

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import numpy as np
import os
import matplotlib.pyplot as plt
conf = ConfigManager("conf.json")

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

    svm = SpeakerClassifier(model='SVM')
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

    svm = SpeakerClassifier(model='SVM')
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

def ubm(save: bool = True, test: bool = False):
    dataset_path = conf.get('database.database_path')
    speakers = conf.get('database.speakers')
    n_components = conf.get('ubm_params.n_components')
    max_iter = conf.get('ubm_params.max_iter')
    random_state = conf.get('ubm_params.random_state')
    save_path = conf.get('ubm_params.save_path')
    inference_data_path = conf.get('ubm_params.inference_data_path')


    dataset_speakers = SpeakerFeatureExtractor(dataset_path, speakers)
    dataset_speakers.features_extraction()

    scaler = RobustScaler()
    mfcc_columns = [col for col in dataset_speakers.df.columns if col.startswith('mfcc')]
    df_scaled = dataset_speakers.df.copy()

    df_scaled[mfcc_columns] = scaler.fit_transform(dataset_speakers.df[mfcc_columns])

    df_train, df_test = train_test_split(
        df_scaled,
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

    gmm_ubm = GMM_UBM(n_components, max_iter, random_state)
    gmm_ubm.train_ubm(df_ubm)
    gmm_ubm.adapt_speaker_models(df_adapt)

    if test:
        n_components_range = range(2, 20)
        bic_scores = []
        aic_scores = []
        for n in n_components_range:
            gmm_ubm = GMM_UBM(n, max_iter, random_state)
            _, bic, aic = gmm_ubm.train_ubm(df_ubm)
            bic_scores.append(bic)
            aic_scores.append(aic)

        # Visualisation des courbes BIC et AIC
        plt.figure(figsize=(8, 6))
        plt.plot(n_components_range, bic_scores, label='BIC', marker='o')
        plt.plot(n_components_range, aic_scores, label='AIC', marker='o')
        plt.xlabel('Nombre de composantes (n_components)')
        plt.ylabel('Score')
        plt.title("Critères BIC et AIC en fonction du nombre de composantes")
        plt.legend()
        plt.grid(True)
        plt.savefig('BIC_AIC.png')
        # plt.show()

    if save:
        np.save(os.path.join(inference_data_path, 'ubm_inference_data.npy'), X_inference)
        np.save(os.path.join(inference_data_path, 'ubm_inference_labels.npy'), y_inference)

        gmm_ubm.save_model(save_path)

    score = 0
    for feature, true_speaker in zip(X_inference, y_inference):
        predictions = gmm_ubm.predict_speaker(feature)
        if predictions == true_speaker:
            score+=1

    print(f"Accuracy: {score/len(y_inference)}")

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

    score = 0
    for feature, speaker in zip(X,y):
        predict_speaker = gmm_ubm.predict_speaker(feature)
        if speaker == predict_speaker:
            score+=1
    print(f'Accuracy: {score/len(y)}')

def train_dnn_ubm(save: bool = True):
    dataset_path = conf.get('database.database_path')
    speakers = conf.get('database.speakers')
    dataset_speakers = SpeakerFeatureExtractor(dataset_path, speakers)
    dataset_speakers.features_extraction()

    df = dataset_speakers.df.copy()

    df_train, df_test = train_test_split(df, test_size=conf.get('dnn_ubm.test_size'),
                                                        random_state=conf.get('dnn_ubm.random_state'))

    df_train, df_inference = train_test_split(df_train, test_size=conf.get('dnn_ubm.test_size'),
                                                        random_state=conf.get('dnn_ubm.random_state'))


    y_train = df_train['speaker']
    y_test = df_test['speaker']
    df_train_encoded = df_train.copy()
    df_test_encoded = df_test.copy()
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_train)
    df_train_encoded['speaker'] = y_encoded
    # Vérification
    df_test_encoded['speaker'] = label_encoder.transform(y_test)

    joblib.dump(label_encoder, os.path.join(conf.get('dnn_ubm.inference_data_path'), 'label_encoder.pkl'))

    if save:
        inference_data_path = conf.get('dnn_ubm.inference_data_path')
        X_inference = np.vstack(df_inference.filter(like='mfcc').values)
        y_inference = np.vstack(df_inference['speaker'].values)
        print(y_inference)
        np.save(os.path.join(inference_data_path, 'dnn_ubm_inference_data.npy'), X_inference)
        np.save(os.path.join(inference_data_path, 'dnn_ubm_inference_labels.npy'), label_encoder.transform(y_inference))


    train_dataset = SpeakerDataLoader(df_train_encoded)
    test_dataset = SpeakerDataLoader(df_test_encoded)
    train_dataloader = DataLoader(train_dataset, batch_size=conf.get('dnn_ubm.batch_size'), shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=conf.get('dnn_ubm.batch_size'), shuffle=True)
    input_dim = train_dataset.features.shape[1]
    n_components = len(speakers)

    model = DNN_UBM(input_dim=input_dim, n_components=n_components)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=conf.get('dnn_ubm.learning_rate'))

    num_epochs = conf.get('dnn_ubm.num_epochs')
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_features, batch_labels in train_dataloader:
            batch_features = torch.tensor(batch_features)

            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_features.size(0)
        avg_loss = epoch_loss / df.size
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_features, batch_labels in test_dataloader:
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            test_loss += loss.item() * batch_features.size(0)

            _, predicted = torch.max(outputs, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

    avg_test_loss = test_loss / len(test_dataloader.dataset)
    accuracy = correct / total
    print(f"Validation Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.4f}")

def dnn_ubm_predict():
    inference_data_path = conf.get('dnn_ubm.inference_data_path')
    speakers = conf.get('database.speakers')
    X_inference = np.load(os.path.join(inference_data_path, 'dnn_ubm_inference_data.npy'))
    y_inference = np.load(os.path.join(inference_data_path, 'dnn_ubm_inference_labels.npy'))
    label_encoder = joblib.load(os.path.join(conf.get('dnn_ubm.inference_data_path'), 'label_encoder.pkl'))

    print(X_inference.shape, y_inference.shape)
    input_dim = X_inference.shape[1]
    n_components = len(speakers)

    model = DNN_UBM(input_dim=input_dim, n_components=n_components)
    correct = 0
    model.eval()
    for x, y in zip(X_inference, y_inference):
        with torch.no_grad():
            # Supposons que x est un vecteur de dimension (input_dim,)
            # On le reformate en (1, input_dim)
            x = x.reshape(1, -1)
            output = model(torch.tensor(x, dtype=torch.float))
            # Pour la classification, récupérer l'index de la classe prédite
            predicted_class = torch.argmax(output, dim=1).item()
            correct += predicted_class == label_encoder.inverse_transform([y])[0]
    print(f"Accuracy: {(correct) / len(y_inference)}")





if __name__ == '__main__':

    train_model()
    weights_path = conf.get("model.save_path")
    infer_model(weights_path)

    # use_clustering()
    ubm(save=True)
    ubm_test()

    # train_dnn_ubm(save=True)
    # dnn_ubm_predict()

