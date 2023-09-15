import pandas as pd

from sklearn.cluster import MeanShift

if __name__ == "__main__":
    dataset = pd.read_csv("./../data/raw/candy.csv")

    print(dataset.head(5))

    X = dataset.drop(["competitorname"], axis = 1)

    meanshift = MeanShift().fit(X)
    labels = meanshift.labels_
    print("Etiquetas :",labels)

    dataset["mean_shift"] =labels
    print(dataset)

    print("Centers mean shift: ",meanshift.cluster_centers_)