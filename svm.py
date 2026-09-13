from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm

cancer_data = datasets.load_breast_cancer()

data = pd.DataFrame(cancer_data.data)
data.columns = cancer_data.feature_names
data["isCancer"] = cancer_data.target

Y = data["isCancer"]
X = data.drop("isCancer",axis = 1)

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,train_size=0.8)

model = svm.SVC(kernel= "linear")
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

print(metrics.accuracy_score(y_test,y_pred))