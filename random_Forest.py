import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("student-mat.csv")

#encode categorical data
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
categorical_cols = ["school","sex","address","famsize","Pstatus","Mjob","Fjob","reason","guardian","schoolsup","famsup","paid","activities","nursery","higher","internet","romantic"]

for col in categorical_cols:
    data[col] = label.fit_transform(data[col])

data.drop(["G1","G2"],axis = 1,inplace = True)
y = data["G3"]
x = data.drop("G3",axis = 1)

#split data set into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,train_size=0.8,random_state=5)

#make model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100,random_state=200)
model.fit(x_train,y_train)

#predict result
y_pred = model.predict(x_test)

#evaluate model using mse
from sklearn.metrics import mean_squared_error,r2_score
mse = mean_squared_error(y_pred=y_pred,y_true=y_test)
r2_score = r2_score(y_test,y_pred)
print(mse)
print(r2_score)

#plot graph
plt.figure(figsize=(10,5))
plt.scatter(range(len(y_test)),y_test,color = "blue")
plt.scatter(range(len(y_pred)),y_pred,color = "red")
plt.title("Actual v Predicted")
plt.xlabel("STudnet Index")
plt.ylabel("acc grade")
plt.show()




