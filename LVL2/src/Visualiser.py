import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, anderson
import numpy as np

class Visualiser:
    def __init__(self, df):
        self.df = df

    def mfcc_histogram(self):
        self.df.filter(like='mfcc').apply(lambda x: plt.hist(x, bins=30, alpha=0.5))
        plt.title("MFCCs distribution")
        plt.show()

    def mfcc_boxplot(self):
        mfcc_matrix = np.stack(self.df.filter(like='mfcc').values)
        plt.figure(figsize=(15, 5))
        sns.boxplot(data=mfcc_matrix)
        plt.title('MFCCs Boxplot')
        plt.show()

    def normality_test(self, test: str):
        mfcc_columns = [col for col in self.df.columns if col.startswith('mfcc_')]

        if test == 'shapiro':
            for col in mfcc_columns:
                stat, p = shapiro(self.df[col])
                print(f"Statistic Shapiro-Wilk: {stat}, p-value: {p}")

                if p > 0.05:
                    print(f"❌ {col} NE suit PAS une distribution normale\n")
                else:
                    print(f"✅ {col} suit une distribution normale\n")

        elif test == 'anderson':
            for col in mfcc_columns:
                result = anderson(self.df[col], dist='norm')
                print(f"📊 Test Anderson-Darling pour {col}:")
                print(f"  Statistique: {result.statistic}")
                print(f"  Seuils critiques: {result.critical_values}")
                print(f"  p-values: {result.significance_level}")

                if result.statistic > result.critical_values[2]:  # Seuil de 5%
                    print(f"❌ {col} NE suit PAS une distribution normale\n")
                else:
                    print(f"✅ {col} suit une distribution normale\n")