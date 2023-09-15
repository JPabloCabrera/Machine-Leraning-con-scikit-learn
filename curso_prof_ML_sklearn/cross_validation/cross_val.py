import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import (
    cross_val_score, KFold
)

if __name__ == "__main__":
    dataset = pd.read_csv("./../data/raw/felicidad.csv")

    X = dataset.drop(["score", "country"], axis = 1)
    y = dataset["score"]

    model = DecisionTreeRegressor()
    score = cross_val_score(model, X, y, cv = 3, scoring = "neg_mean_squared_error")
    print(score)
    print(np.abs(np.mean(score)))

    kf = KFold(n_splits = 4, random_state = 42, shuffle = True)
    for train, test in kf.split(X, y):
        print(train, test)


        
