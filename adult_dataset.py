import pandas as pd
data = pd.read_csv("adult (1).csv")
data.columns = ("age","workclass","fnlwgt","education","education_num","marital_status","occupation","relationship","race","sex","capital_gain","capital_loss","hours_per_week","native_country","income")

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
catergorical = ["workclass","education","marital_status","occupation","relationship","race","sex","native_country","income"]
for col in catergorical:
    data[col]=label_encoder.fit_transform(data[col])

data.drop("capital_loss",axis = 1,inplace=True)
Y = data["capital_gain"]
X = data.drop("capital_gain",axis = 1)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,train_size=0.8,random_state=5)

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=200,random_state=100)
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

from sklearn.metrics import mean_squared_error,r2_score
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

print(mse)
print(r2)

import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
plt.scatter(range(len(y_pred)),y_pred,color = "purple")
plt.scatter(range(len(y_test)),y_test,color = "blue")
plt.title("Actual v Predicted")
plt.xlabel("index")
plt.ylabel("capital gain")
plt.show()
