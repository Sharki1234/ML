import pandas

data = pandas.read_csv(r"C:\Users\rakhi\AppData\Local\Temp\198880f8-6d3c-49e8-b651-7fa31c433095_archive (1).zip.095\train.csv")

y = data["blue"]
data.drop("blue",axis = 1,inplace = True)
x = data

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,train_size=0.8)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

import seaborn as sns

from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
matrix = confusion_matrix(y_test,y_pred)
sns.heatmap(matrix,annot=True,fmt="d")
plt.title("Matrix")
plt.xlabel("pred")
plt.ylabel("real")
plt.show()



