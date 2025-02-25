import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA

class Visualiser:
    def __init__(self):
        pass

    def draw_ellipse(self, position, covariance, ax, **kwargs):
        """Dessine une ellipse représentant une composante gaussienne en 2D."""
        if covariance.shape == (2, 2):
            U, s, _ = np.linalg.svd(covariance)
            angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
            width, height = 2 * np.sqrt(s)
        else:
            width = height = 2 * np.sqrt(covariance)
            angle = 0
        ellipse = Ellipse(xy=position, width=width, height=height, angle=angle, **kwargs)
        ax.add_patch(ellipse)

    def plot_gmm_ubm(self, gmm, X, title="UBM Visualization"):
        """
        Applique une PCA pour projeter X en 2D et affiche :
          - Les données (scatter)
          - Les ellipses représentant chaque composante du GMM
        """
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)

        # Transformer les moyennes et covariances dans l'espace PCA
        means_2d = pca.transform(gmm.means_)

        # Pour la covariance, on doit faire une transformation :
        # Pour chaque composante, projeter la covariance : C_2d = P * C * P.T
        covariances_2d = []
        for cov in gmm.covariances_:
            cov_2d = pca.components_ @ cov @ pca.components_.T
            covariances_2d.append(cov_2d)

        fig, ax = plt.subplots(figsize=(8, 6))

        # Afficher les données projetées
        ax.scatter(X_2d[:, 0], X_2d[:, 1], s=30, alpha=0.5, cmap='viridis')

        # Afficher chaque composante
        for mean, cov in zip(means_2d, covariances_2d):
            self.draw_ellipse(mean, cov, ax, alpha=0.5, edgecolor='red', facecolor='none', lw=2)
            ax.scatter(mean[0], mean[1], c='red', s=100, marker='x')

        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        plt.show()