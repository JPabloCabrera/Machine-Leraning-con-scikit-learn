import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


if __name__ == "__main__":

    dt_heart = pd.read_csv("./../data/raw/heart.csv")

    print(dt_heart["target"].describe())

    X = dt_heart.drop(["target"], axis = 1)

    y = dt_heart["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.35)

    knn_class = KNeighborsClassifier().fit(X_train, y_train)
    knn_predict = knn_class.predict(X_test)
    print("="*64)
    print("KNeighborsClassifier:\n", accuracy_score(y_test, knn_predict))
    
    bagging_class = BaggingClassifier(base_estimator = KNeighborsClassifier(), n_estimators = 50).fit(X_train, y_train)
    bagging_predict = bagging_class.predict(X_test)
    print("="*64)
    print("BaggingClassifier con KNeighborsClassifier:\n",accuracy_score(y_test, bagging_predict))
    
   
    