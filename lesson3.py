import pandas as pd
import  numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(r"C:\Users\rakhi\OneDrive\Desktop\ML\titanic(2).csv")

from sklearn import preprocessing

label_encode = preprocessing.LabelEncoder()
data["Sex"] = label_encode.fit_transform(data["Sex"])
y = data["Pclass"]
data.drop("Pclass",axis = 1,inplace=True)
data.drop("Name",axis = 1,inplace=True)
x = data


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,train_size=0.8)

from sklearn.linear_model import LogisticRegression

lg = LogisticRegression()
lg.fit(x_train,y_train)
y_pred = lg.predict(x_test)

from sklearn.metrics import classification_report,confusion_matrix
matrix = confusion_matrix(y_test,y_pred)
sns.heatmap(matrix,annot=True,fmt="d")
plt.title("Matrix")
plt.xlabel("pred")
plt.ylabel("real")
plt.show()

print(classification_report(y_pred=y_pred,y_true = y_test))