import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(r"car.csv")
data.columns = ("sales","maintenence","doors","persons","boot_space","safety","class")

from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
data["sales"] = label.fit_transform(data["sales"])
data["maintenence"] = label.fit_transform(data["maintenence"])
data["doors"] = label.fit_transform(data["doors"])
data["persons"] = label.fit_transform(data["persons"])
data["boot_space"] = label.fit_transform(data["boot_space"])
data["safety"] = label.fit_transform(data["safety"])
data["class"] = label.fit_transform(data["class"])

y = data["class"]
data.drop("class",axis = 1,inplace=True)
x = data

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,train_size=0.8,random_state=5)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion="entropy",random_state=0)
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)
from sklearn.metrics import confusion_matrix,classification_report
matrix = confusion_matrix(y_test,y_pred)
sns.heatmap(matrix,annot=True,fmt="d")
plt.title("Matrix")
plt.xlabel("pred")
plt.ylabel("real")
plt.show()

print(classification_report(y_true=y_test,y_pred=y_pred))
