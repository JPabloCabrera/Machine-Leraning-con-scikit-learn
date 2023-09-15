import pandas as pd

import sklearn

from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import HuberRegressor

from sklearn.svm import SVR

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":

    dataset = pd.read_csv("./../data/raw/felicidad_corrupt.csv")

    print(dataset.head(5))

    X = dataset.drop(["country", "score"], axis = 1)
    y = dataset["score"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = 0.3)

    estimadores = {
        "SVR" : SVR(gamma = "auto", C = 10, epsilon = 0.1),
        "RANSAC" : RANSACRegressor(),
        "HuberRegressor" : HuberRegressor(epsilon = 1.35)
    }

    for name, estimador in estimadores.items():
        estimador.fit(X_train, y_train)
        y_predict = estimador.predict(X_test)

        print("="*64)
        print(name)
        print("MSE :", mean_squared_error(y_test, y_predict))

    
