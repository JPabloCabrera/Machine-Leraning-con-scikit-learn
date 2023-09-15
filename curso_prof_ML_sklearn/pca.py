import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

from sklearn.decomposition import PCA 
from sklearn.decomposition import IncrementalPCA 

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



if __name__ == "__main__":

    df_heart = pd.read_csv("./data/raw/heart.csv")
    print(df_heart.head(5))

    df_features = df_heart.drop(["target"], axis = 1)
    df_target = df_heart["target"]

    df_features = StandardScaler().fit_transform(df_features)

    X_train, X_test, y_train, y_test = train_test_split(df_features, df_target, test_size = 0.3, random_state = 42)
    
    print(X_train.shape)
    print(y_train.shape)

    # n components
    n = 3
    pca = PCA(n_components = n)
    pca.fit(X_train)

    ipca = IncrementalPCA(n_components = n, batch_size = 10)
    ipca.fit(X_train)

    plt.plot(np.arange(len(pca.explained_variance_)),pca.explained_variance_ratio_)
    plt.plot(np.arange(len(ipca.explained_variance_)),ipca.explained_variance_ratio_, "rx-")
    plt.show()

    dt_train = pca.transform(X_train)
    dt_test = pca.transform(X_test)

    logistic = LogisticRegression(solver = "lbfgs")
    logistic.fit(dt_train, y_train)
    print("PCA Score: ", logistic.score(dt_test,y_test))

    dt_train = ipca.transform(X_train)
    dt_test = ipca.transform(X_test)
    
    logistic = LogisticRegression(solver = "lbfgs")
    logistic.fit(dt_train, y_train)
    print("IPCA score: ", logistic.score(dt_test, y_test))
