import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(r"C:\Users\rakhi\Downloads\iris_data.csv")

y = data["species"]
data.drop("species",axis = 1,inplace= True)
x = data

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,test_size=0.2)

from sklearn.preprocessing import StandardScaler,LabelEncoder
scaler = StandardScaler()
scaler.fit_transform(x_train)
label_encoder = LabelEncoder()
label_encoder.fit_transform(y_train)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train,y_train)

scaler.transform(x_test)
label_encoder.transform(y_test)
y_pred = classifier.predict(x_test)

from sklearn.metrics import classification_report,confusion_matrix
matrix = confusion_matrix(y_true=y_test,y_pred = y_pred)
sns.heatmap(matrix,annot= True,fmt = "d")
plt.title("Matrix")
plt.xlabel("pred")
plt.ylabel("real")
plt.show()

