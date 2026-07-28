def findMean(list):
    return sum(list)/len(list)
x = [1,2,3,4,5]
y = [1,3,3,4,5]

meanx = findMean(x)
meany = findMean(y)
num = 0
den = 0
for i in range(len(x)):
    num = num +(( x[i] - meanx )* (y[i] -meany))
    den = den + (x[i] - meanx)**2
m = num/den
c = meany - m * meanx
print(m,c)

import numpy as np
from  sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

x = np.array([[1],[2],[3],[4],[5]])
y = np.array([[1],[3],[3],[4],[5]])

reg = LinearRegression().fit(x,y)
print(reg.coef_,reg.intercept_)

plt.scatter(x, y, color='blue', label='Data Points')

plt.plot(y, reg.predict(x), color='red', label='Best Fit Line')

plt.xlabel('x')

plt.ylabel('y')

plt.legend()

plt.show()