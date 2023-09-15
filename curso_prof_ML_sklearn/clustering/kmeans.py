import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import MiniBatchKMeans

if __name__ == "__main__":

    dataset = pd.read_csv("./../data/raw/candy.csv")
    print(dataset.head())

    X = dataset.drop("competitorname", axis = 1)

    kmeans = MiniBatchKMeans(n_clusters = 4, batch_size = 8).fit(X)
    kmeans_predict = kmeans.predict(X)
    print(kmeans_predict)
    print(len(kmeans.cluster_centers_))

    dataset["group"] = kmeans_predict

    