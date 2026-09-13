import pandas as pd
data = pd.read_csv(f"processed.cleveland.data",header=None)
data.columns = ("age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","target")


Y = (data["target"]>0).astype(int)
X = data.drop("target",axis = 1)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.1,random_state=42,stratify=Y)

from sklearn import svm
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler
# x_train_scaler = scaler.fit_transform(x_train)
# x_test_scaler = scaler.fit_transform(x_test)


model = svm.SVC(kernel="linear")
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))