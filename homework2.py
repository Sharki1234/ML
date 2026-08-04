import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data = pd.read_csv(r"C:\Users\rakhi\Downloads\data (2).csv")
x = data[["sepal_length","sepal_width"]]
y = data["petal_width"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,train_size=0.8,random_state=5)
model = LinearRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

rmse = mean_squared_error(y_test,y_pred)
print(rmse)

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)

model_p = LinearRegression()
model_p.fit(x_train_poly,y_train)
y_pred_poly = model_p.predict(x_test_poly)
rmse_poly = mean_squared_error(y_test,y_pred_poly)
print(rmse_poly)