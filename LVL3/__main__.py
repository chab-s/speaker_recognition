from src import SpeakerFeatureExtractor
from src import ConfigManager
from src import GMM_UBM

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import numpy as np
import os
import matplotlib.pyplot as plt
conf = ConfigManager("conf.json")

def ubm(save: bool = True, test: bool = False):
    dataset_path = conf.get('database.database_path')
    speakers = conf.get('database.speakers')
    n_components = conf.get('gmm_ubm.n_components')
    max_iter = conf.get('gmm_ubm.max_iter')
    random_state = conf.get('gmm_ubm.random_state')
    save_path = conf.get('gmm_ubm.save_path')
    inference_data_path = conf.get('gmm_ubm.inference_data_path')


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
    n_components = conf.get('gmm_ubm.n_components')
    max_iter = conf.get('gmm_ubm.max_iter')
    random_state = conf.get('gmm_ubm.random_state')
    save_path = conf.get('gmm_ubm.save_path')
    inference_data_path = conf.get('gmm_ubm.inference_data_path')

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

if __name__ == '__main__':
    ubm(save=True)
    ubm_test()






