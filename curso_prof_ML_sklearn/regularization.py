import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from sklearn.linear_model import ElasticNet

from sklearn.metrics import mean_squared_error

if __name__ == "__main__":

    dataset = pd.read_csv("./data/raw/felicidad.csv")

    print(dataset.info())


    X = dataset[["gdp", "family", "lifexp", "freedom", "generosity" ,"corruption", "dystopia"]]
    y = dataset["score"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

    print(X_train.shape)
    print(y_train.shape)

    modelLinear = LinearRegression().fit(X_train, y_train)
    y_predict_linear = modelLinear.predict(X_test)

    modelLasso = Lasso(alpha = 0.02).fit(X_train, y_train)
    y_predict_lasso = modelLasso.predict(X_test)

    modelRidge = Ridge(alpha = 1).fit(X_train, y_train)
    y_predict_ridge = modelRidge.predict(X_test)



    modelElasticNet = ElasticNet(alpha = 0.0002).fit(X_train, y_train)
    y_predict_elasticnet = modelElasticNet.predict(X_test)



    linear_loss = mean_squared_error(y_test, y_predict_linear)
    print("Linear loss: ",linear_loss)

    lasso_loss = mean_squared_error(y_test, y_predict_lasso)
    print("Lasso loss: ", lasso_loss)

    ridge_loss = mean_squared_error(y_test, y_predict_ridge)
    print("Ridge loss: ", ridge_loss)

    elasticnet_loss = mean_squared_error(y_test, y_predict_elasticnet)
    print("ElasticNet loss: ", elasticnet_loss)


    print("="*32)
    print("Lasso model coef",modelLasso.coef_)

    print("="*32)
    print("Ridge coefs: ", modelRidge.coef_)

    print("="*32)
    print("ElasticNet coefs: ", modelElasticNet.coef_)


    
    