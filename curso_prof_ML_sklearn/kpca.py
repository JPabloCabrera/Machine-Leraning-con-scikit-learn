import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.decomposition import KernelPCA

if __name__ == "__main__":
    
    df_heart = pd.read_csv("./data/raw/heart.csv")

    df_features = df_heart.drop(["target"], axis = 1)
    df_target = df_heart["target"]

    df_features = StandardScaler().fit_transform(df_features)

    X_train, X_test, y_train, y_test = train_test_split(df_features, df_target, test_size = 0.3, random_state = 42)

    scores = []
    n = np.arange(1,8)

    for _ in n:
        kpca = KernelPCA(n_components = _, kernel ="poly")
        kpca.fit(X_train)

        dt_train = kpca.transform(X_train)
        dt_test = kpca.transform(X_test)

        logistic = LogisticRegression(solver = "lbfgs")
        logistic.fit(dt_train, y_train)
        #print("Score to kernelPCA data: ", logistic.score(dt_test, y_test))

        scores.append(logistic.score(dt_test, y_test))


    plt.plot(range(len(scores)), scores)
    plt.show()


