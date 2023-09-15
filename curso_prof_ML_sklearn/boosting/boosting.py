import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


if __name__ == "__main__":

    dt_heart = pd.read_csv("./../data/raw/heart.csv")

    print(dt_heart["target"].describe())

    X = dt_heart.drop(["target"], axis = 1)

    y = dt_heart["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.35)

    boosting = GradientBoostingClassifier(n_estimators = 50).fit(X_train, y_train)
    boosting_predict = boosting.predict(X_test)
    print("="*64)
    print("GradientBoostingClassifier accuracy score")
    print(accuracy_score(y_test,boosting_predict))