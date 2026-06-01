# Try classifying classes 1 and 2 from the iris dataset with SVMs, with the 2 first features.
# Leave out 10% of each class and test prediction performance on these observations.
from sklearn import svm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

#load iris dataset
iris = load_iris()
x = iris.data
y = iris.target #already encoded target label


#extract first two features of iris data set and classify on class labels 1 and 2

y_2c = y[y>0] #all the class entries above 0
x_2f = x[:,:2] #extracts all the rows from the first two features
x_2f = x_2f[y>0] #extracts the instances corresponding to class 1 and class 2

#train_test split on data

x_train, x_test, y_train, y_test = train_test_split(x_2f,y_2c, test_size=0.1, random_state=42)


#Train SVM classifier on the above dataset
clf_poly = svm.SVC(kernel='poly', degree=3, C=1)
clf_poly.fit(x_train,y_train)
clf_rbf = svm.SVC(kernel='rbf', gamma=0.5, C=1)
clf_rbf.fit(x_train,y_train)

#prediction
from sklearn.metrics import accuracy_score

y_pred_rbf = clf_rbf.predict(x_test)
y_pred_poly = clf_poly.predict(x_test)

print("RBF Kernel Accuracy:", accuracy_score(y_test, y_pred_rbf))
print("Polynomial Kernel Accuracy:", accuracy_score(y_test, y_pred_poly))



import numpy as np
import matplotlib.pyplot as plt


def plot_decision_boundary(clf, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.title(title)
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plot_decision_boundary(clf_poly, x_train, y_train, "Polynomial Kernel (Degree=3)")
plt.subplot(1, 2, 2)
plot_decision_boundary(clf_rbf, x_train, y_train, "RBF Kernel (Gamma=0.5)")
plt.show()