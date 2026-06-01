#OJdataset
import numpy as np
from ISLP import load_data
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import svm



#load OJ dataset
data = load_data('OJ')
print(data)
print(data.keys())
print(data.head(5))


#extract x and y from the data
#treat purchase column as the target i.e y
x = data.drop(columns=['Purchase'])
y = data['Purchase']

#label encoding as y is a categorical value
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
# print(y)

#feature encoding
#as the values are boolean we use label encoding
x_store = x['Store7']
feature_encoder = LabelEncoder()
feature = feature_encoder.fit_transform(x_store)
print(feature.shape)

#add the encoded store7 in the design matrix x
x['Store7'] = feature

print(x['Store7'])

#Part-A)
#split the dataset into 1000 train set
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=1000, random_state=42)
# print(x_train.shape)
# print(x_test.shape)

#fit a support vector classifier to the given data
svc_clf = svm.LinearSVC(C=0.01)
svc_clf.fit(x_train,y_train)

#predict on x_test
y_pred = svc_clf.predict(x_test)

##accuracy
#test accuracy
accuracy_train = accuracy_score(y_test,y_pred)
print(accuracy_train)

#train accuracy
#same way

#c) GridSearchCV

from sklearn.model_selection import GridSearchCV

# c_values = np.linspace(0.01,10,10) ##taking values from 0.01 to 10
# If your data benefits from weaker regularization (more flexibility), a larger C (like 2.23) from linspace might be better.
c_values = np.logspace(-2,1,10)
# If your data benefits from stronger regularization (to prevent overfitting), a smaller C (like 0.1) from logspace will win.

parameters = {
              'C':c_values}
clf = GridSearchCV(svc_clf, parameters)
clf.fit(x_train, y_train)
print("Best parameters:", clf.best_params_)
