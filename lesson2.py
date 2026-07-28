import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

california = fetch_california_housing()
dataframe = pd.DataFrame(california.data,columns = california.feature_names)
dataframe["MedHouseVal"] = california.target

X_data = dataframe[["MedInc","AveRooms"]]
Y_data = dataframe["MedHouseVal"]

X_train,X_test,Y_train,Y_test =train_test_split(X_data,Y_data,test_size=0.2,train_size=0.8,random_state=5)

ln = LinearRegression()
ln.fit(X_train,Y_train)
y_predictions = ln.predict(X_test)

rmse = np.sqrt(mean_squared_error(Y_test,y_predictions))
print(rmse)

poly = PolynomialFeatures(degree = 2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly,Y_train)
y_pred_poly = poly_model.predict(X_test_poly)
rmse_poly = np.sqrt(mean_squared_error(Y_test,y_pred_poly))
print(rmse_poly)