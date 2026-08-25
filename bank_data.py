import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("bank-full.csv",sep=";")


from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
data["job"] = label.fit_transform(data["job"])
data["marital"] = label.fit_transform(data["marital"])
data["education"] = label.fit_transform(data["education"])
data["housing"] = label.fit_transform(data["housing"])
data["loan"] = label.fit_transform(data["loan"])
data["contact"] = label.fit_transform(data["contact"])
data["month"] = label.fit_transform(data["month"])
data["default"] = label.fit_transform(data["default"])
data["poutcome"] = label.fit_transform(data["poutcome"])
data["y"] = label.fit_transform(data["y"])

y = data["y"]
data.drop("y",axis = 1,inplace=True)
x = data

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,train_size=0.8)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion="entropy",random_state=2)
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)

import seaborn as sns
from sklearn.metrics import confusion_matrix,classification_report
Matrix = confusion_matrix(y_test,y_pred)
sns.heatmap(Matrix,annot=True,fmt = "d")
plt.title("Matrix")
plt.xlabel("pred")
plt.ylabel("real")
plt.show()
print(classification_report(y_test,y_pred))