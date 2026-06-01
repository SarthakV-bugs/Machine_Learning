#part b question 1
import numpy as np
from matplotlib import pyplot as plt
from pandas.core.common import random_state
from pandas.io.sas.sas_constants import dataset_length
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sympy.printing.pretty.pretty_symbology import line_width

#a)
np.random.seed(1)

x = np.random.normal(loc=0,scale=1,size =200) #feature one
# x = x.reshape(-1,1)
# print(x.reshape(-1,1))
#b)
e = np.random.normal(loc=0, scale=0.25, size=200) #noise vector e
print(e)

#c)
# generate y as per the model y = -1.1 + 0.6x + e
y = -1.1 + 0.6 * x + e #if you reshape x here, then y is generated as (200,200)
# Your shape mismatch is due to NumPy broadcasting a (200,1) array plus a (200,) array.
# Fix by making both arrays 1D or both 2D with compatible shapes.
print(y)

print(y.shape)

#values of theta0 = -1.1 and theta 1 is 0.6


#plot scatter
# plt.scatter(x,y,c='viridis')
# plt.title('X v/s Y' )
# plt.xlabel('x values')
# plt.ylabel('y values')
# plt.show()

#y is linearly dependent on the values of x fits the equation properly

#d)
# fit a linear regression model on the provided dataset
x = x.reshape(-1,1) #tricky guy
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=42)

#use linear regression model


model =  LinearRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

#evaluate r2score

r2 = r2_score(y_test,y_pred)
print(r2)


print("Intercept (theta0):", model.intercept_)
print("Coefficient (theta1):", model.coef_)


#e)

#sort the data for a smooth line, very important
sort_idx = np.argsort(x_test.flatten())

plt.scatter(x_test[sort_idx],y_pred[sort_idx],color='blue', edgecolors='k') #to plot the data points
plt.plot(x_test[sort_idx],y_pred[sort_idx],color='red')
plt.title("xtest vs ytest")
plt.xlabel("x_test")
plt.ylabel("y_test")
plt.show()


