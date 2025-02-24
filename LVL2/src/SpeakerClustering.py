from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
import numpy as np
from src import ConfigManager
from sklearn.metrics import adjusted_rand_score, homogeneity_score, completeness_score
import matplotlib.pyplot as plt

conf = ConfigManager('conf.json')

class SpeakerClustering:
    def __init__(self, model: str, n_clusters: int):
        self.n_clusters = n_clusters

        if model == 'KMeans':
            self.model = KMeans(n_clusters=self.n_clusters, random_state=conf.get('model.random_state'))
        elif model == 'BanditPam':
            self.model = KMedoids(n_clusters=self.n_clusters, random_state=conf.get('model.random_state'))

    def test_scaler(self, mfcc):
        scalers = {
            "StandardScaler": StandardScaler(),
            "MinMaxScaler": MinMaxScaler(),
            "RobustScaler": RobustScaler(),
            "PowerTransformer": PowerTransformer(method='yeo-johnson')
        }

        for name, scaler in scalers.items():
            X_scaled = scaler.fit_transform(mfcc)
            plt.figure(figsize=(10, 4))
            plt.hist(X_scaled.flatten(), bins=30, alpha=0.5, label=name)
            plt.title(f"Distribution après {name}")
            plt.legend()
            plt.show()

    def clusterization(self, X):
        scaler = RobustScaler()  # PowerTransformer(method='yeo-johnson')
        X_scaled = scaler.fit_transform(X)
        labels = self.model.fit_predict(X_scaled)
        return labels

    def evaluate_clusters(self, label, cluster):
        ari_score = adjusted_rand_score(label, cluster)
        homogeneity = homogeneity_score(label, cluster)
        completeness = completeness_score(label, cluster)

        print(f"ARI score: {ari_score:.4f}")
        print(f"homogeneity: {homogeneity:.4f}")
        print(f"Completeness: {completeness:.4f}")

        return {"ARI": ari_score, "Homogeneity": homogeneity, "Completeness": completeness}




