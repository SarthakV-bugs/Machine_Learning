import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score
from ISLP import load_data


# Load the dataset
#df = pd.read_csv("OJ.csv")  # Replace with correct path if necessary
df = load_data('OJ')

# Encode categorical variables
df['Purchase'] = LabelEncoder().fit_transform(df['Purchase'])  # CH=1, MM=0

# Define X and y
X = df.drop(columns=['Purchase'])
y = df['Purchase']

# Encode remaining categorical features
X = pd.get_dummies(X, drop_first=True)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# (a) Split data: 1000 training samples, rest for testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=1000, random_state=42, stratify=y)

# (b) Train LinearSVC with C=0.01
svc_linear = LinearSVC(C=0.01, max_iter=10000)
svc_linear.fit(X_train, y_train)
y_train_pred = svc_linear.predict(X_train)
y_test_pred = svc_linear.predict(X_test)

acc_train_linear = accuracy_score(y_train, y_train_pred)
acc_test_linear = accuracy_score(y_test, y_test_pred)

print(f"(b) Linear SVC with C=0.01 -> Train Accuracy: {acc_train_linear:.3f}, Test Accuracy: {acc_test_linear:.3f}")

# (c) Use GridSearchCV to find best C
param_grid = {'C': np.linspace(0.01, 10, 50)}
grid_search = GridSearchCV(LinearSVC(max_iter=10000), param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_C = grid_search.best_params_['C']
print(f"(c) Best C from GridSearchCV: {best_C}")

# (d) Train LinearSVC with best C
svc_linear_best = LinearSVC(C=best_C, max_iter=10000)
svc_linear_best.fit(X_train, y_train)
y_train_best = svc_linear_best.predict(X_train)
y_test_best = svc_linear_best.predict(X_test)

acc_train_best = accuracy_score(y_train, y_train_best)
acc_test_best = accuracy_score(y_test, y_test_best)

print(f"(d) Linear SVC with optimal C={best_C} -> Train Accuracy: {acc_train_best:.3f}, Test Accuracy: {acc_test_best:.3f}")

# (e) SVM with RBF kernel (default gamma)
svc_rbf = SVC(kernel='rbf')
svc_rbf.fit(X_train, y_train)
y_train_rbf = svc_rbf.predict(X_train)
y_test_rbf = svc_rbf.predict(X_test)

acc_train_rbf = accuracy_score(y_train, y_train_rbf)
acc_test_rbf = accuracy_score(y_test, y_test_rbf)

print(f"(e) SVM with RBF Kernel -> Train Accuracy: {acc_train_rbf:.3f}, Test Accuracy: {acc_test_rbf:.3f}")

# Summary
print("\nSummary of all models:")
print(f"Linear SVC (C=0.01): Train={acc_train_linear:.3f}, Test={acc_test_linear:.3f}")
print(f"Linear SVC (best C={best_C}): Train={acc_train_best:.3f}, Test={acc_test_best:.3f}")
print(f"SVM RBF: Train={acc_train_rbf:.3f}, Test={acc_test_rbf:.3f}")
