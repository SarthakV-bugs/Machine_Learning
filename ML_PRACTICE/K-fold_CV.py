# # 1.K-fold cross validation. Implement for K = 10. Implement from scratch, then, use
# # scikit-learn methods.
# # 2.Data normalization - scale the values between 0 and 1. Implement code from scratch.
# # 4.Data standardization - scale the values such that mean of new dist = 0 and sd = 1.
# # Implement code from scratch.
# # 5.Use validation set to do feature and model selection.
# import random
# from sklearn.linear_model import LinearRegression
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
#
# from sklearn.metrics import r2_score
#
# ###K-fold cross validation from scratch
#
# #step1: load dataset
#
# data = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
# x= data.drop(columns=["disease_score_fluct", "disease_score"])
# y =data["disease_score_fluct"]
#
# ##Step2: k-fold logic
#
# #shuffle the indices of the sample dataset to prevent any biases
#
# def shuffle_datasets(x,y,seed=None):
#
#     if seed is not None:
#         random.seed(seed)
#     n = len(x) #length of the dataset
#     indices = list(range(n)) #makes a list of indices equal to that of the length of dataset
#     for i in range(n-1,0,-1): #iterate from end to front
#         j = random.randint(0,i) #pick random index from 0 to i
#         #shuffling logic goes here
#         indices[i], indices[j] = indices[j], indices[i]
#
#     # apply indices on X and Y to create X_shuffled and Y_shuffled to maintain the corresponding x and y order
#     # Using Python lists (slower)
#     # X_shuffled_list = [X[i] for i in indices]
#     # y_shuffled_list = [y[i] for i in indices]
#
#     x_shuffled = np.array(x)[indices]
#     y_shuffled = np.array(y)[indices]
#
#     return x_shuffled, y_shuffled
#
# def k_fold_split(x_shuffled,y_shuffled,k_folds):
#     ##k_folds -> no of folds to be created
#     ##split the dataset into k folds
#
#     fold_size = len(x_shuffled) // k_folds #gives the size of each fold
#     folds =[]
#
#     for i in range(k_folds):
#         start, end = i*fold_size, (i+1)*fold_size
#         x_fold = x_shuffled[start:end] #6 sample in each fold
#         y_fold = y_shuffled[start:end] #6 target values corresponding to it in each fold
#         folds.append((x_fold,y_fold))
#
#     for i, fold in enumerate(folds, start=1):
#         print(f"Fold {i}:")
#         print(f"X:\n{fold[0]}")
#         print(f"Y:\n{fold[1]}")
#         print("-" * 30)
#
#     return folds
# #
# # def train_model(x_train,y_train):
# #
# #     print("--------------Training---------------")
# #     model = LinearRegression()
# #
# #     #train the model on x_train and y_train
# #     model.fit(x_train,y_train)
# #
# #     return model
#
# ####code block to perform k_fold cross validation on all the k_folds
#
# def k_fold_cv(model,folds):
#
#     """In this code block, we use one fold as x_test and the rest are used for training the model.
#     The score is computed for each fold and stored in a score list"""
#
#     score = []
#     for i in range(len(folds)): #iterates through each fold
#         x_test_cv,y_test_cv = folds[i] #one fold is assigned to test
#         # print(f"x_test_cv: \n{x_test_cv}")
#
#         ##x_train_cv should consist of all the folds - test fold
#         x_train_cv = [folds[j][0] for j in range(len(folds)) if j != i]
#         # print(x_train_cv)
#         y_train_cv = [folds[j][1] for j in range(len(folds)) if j != i]
#
#         x_train_cv = np.vstack(x_train_cv)  #the entries in the list are vertically stacked in a matrix format
#         y_train_cv = np.hstack(y_train_cv) #single array
#
#
#         #train the model for this fold set
#         model.fit(x_train_cv,y_train_cv)
#
#         #predict the y test using x test
#         y_pred = model.predict(x_test_cv)
#         # print(f"y_pred: \n{y_pred}")
#
#         #r2score computation
#         r2 = r2_score( y_test_cv, y_pred)
#         #append the score for the fold
#         score.append(r2)
#
#
#     return score
#
#
#
#
#
#
#
#
# def main():
#
#     x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)
#     x_shuffled, y_shuffled = shuffle_datasets(x_train,y_train,seed=42)
#     # print(x_shuffled)
#     ##code to create the splits
#     folds = k_fold_split(x_shuffled,y_shuffled,k_folds=4) ###to prevent the 42 training data problem, we can use k as 6,8,4,etc.
#     # print(folds)
#
#     model = LinearRegression()
#
#     cv_score= k_fold_cv(model,folds)
#     for i, score in enumerate(cv_score, start=1):
#         print(f"Fold{i}: R2 Score = {score:.4f}")
#
#     # Compute and print mean and standard deviation
#     avg_score = np.mean(cv_score)
#     std_dev = np.std(cv_score)
#
#     print(f"\nAverage R² Score: {avg_score:.4f}")
#     print(f"Standard Deviation of R² Scores: {std_dev:.4f}")
#
#
# if __name__ == '__main__':
#     main()

import random
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

### K-fold cross-validation from scratch

# Step 1: Load dataset
data = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
X = data.drop(columns=["disease_score_fluct", "disease_score"])
y = data["disease_score_fluct"]


# Step 2: Shuffle the dataset
def shuffle_datasets(X, y, seed=None):
    if seed is not None:
        random.seed(seed)
    indices = list(range(len(X)))
    random.shuffle(indices)  # Shuffling indices

    X_shuffled = np.array(X)[indices]
    y_shuffled = np.array(y)[indices]
    return X_shuffled, y_shuffled


# Step 3: K-Fold splitting (Fixed version)
def k_fold_split(X_shuffled, y_shuffled, k_folds):
    fold_size = len(X_shuffled) // k_folds
    fold_sizes = [fold_size] * k_folds

    # Distribute remaining samples across some folds
    for i in range(len(X_shuffled) % k_folds):
        fold_sizes[i] += 1

    folds = []
    start_idx = 0
    for size in fold_sizes:
        X_fold = X_shuffled[start_idx:start_idx + size]
        y_fold = y_shuffled[start_idx:start_idx + size]
        folds.append((X_fold, y_fold))
        start_idx += size

    return folds


# Step 4: Perform K-Fold Cross-Validation
def k_fold_cv(model, folds):
    scores = []

    for i in range(len(folds)):
        X_test_cv, y_test_cv = folds[i]  # Test set

        # Train set: All folds except the test fold
        X_train_cv = np.vstack([folds[j][0] for j in range(len(folds)) if j != i])
        y_train_cv = np.hstack([folds[j][1] for j in range(len(folds)) if j != i])

        model.fit(X_train_cv, y_train_cv)
        y_pred = model.predict(X_test_cv)

        # Compute R² Score
        r2 = r2_score(y_test_cv, y_pred)
        scores.append(r2)
        print(f"Fold {i + 1}: R² Score = {r2:.4f}")

    # Compute mean and standard deviation
    avg_score = np.mean(scores)
    std_dev = np.std(scores)

    print(f"\nAverage R² Score: {avg_score:.4f}")
    print(f"Standard Deviation of R² Scores: {std_dev:.4f}")

    return scores


# Step 5: Normalization from scratch (Min-Max Scaling)
def normalize(X):
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    return (X - X_min) / (X_max - X_min)


# Step 6: Standardization from scratch (Z-score Scaling)
def standardize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std


# Main function
def main():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Shuffle dataset
    X_shuffled, y_shuffled = shuffle_datasets(X_train, y_train, seed=42)

    # Normalize and standardize the data
    X_normalized = normalize(X_shuffled)
    X_standardized = standardize(X_shuffled)

    # Perform K-Fold Cross-Validation
    k_folds = 6
    folds = k_fold_split(X_standardized, y_shuffled, k_folds)
    model = LinearRegression()
    k_fold_cv(model, folds)


if __name__ == "__main__":
    main()

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.feature_selection import RFE

# Step 1: Load dataset
data = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
X = data.drop(columns=["disease_score_fluct", "disease_score"])
y = data["disease_score_fluct"]

# Step 2: Split into Train (70%), Validation (15%), Test (15%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}, Test size: {len(X_test)}")


# Step 3: Normalize (Min-Max Scaling)
def normalize(X):
    return (X - X.min()) / (X.max() - X.min())


X_train = normalize(X_train)
X_val = normalize(X_val)
X_test = normalize(X_test)


# Step 4: Feature Selection (RFE with Linear Regression)
def select_features(X_train, y_train):
    model = LinearRegression()
    rfe = RFE(model, n_features_to_select=5)  # Select top 5 features
    X_train_rfe = rfe.fit_transform(X_train, y_train)
    selected_features = X_train.columns[rfe.support_]
    print(f"Selected Features: {list(selected_features)}")
    return X_train_rfe, selected_features


X_train_rfe, selected_features = select_features(X_train, y_train)
X_val_rfe = X_val[selected_features]
X_test_rfe = X_test[selected_features]

# Step 5: Compare Models on Validation Set
models = {
    "Linear Regression": LinearRegression(),
    "Lasso Regression": Lasso(alpha=0.1),
    "Ridge Regression": Ridge(alpha=1.0)
}

best_model = None
best_score = float("-inf")

for name, model in models.items():
    model.fit(X_train_rfe, y_train)
    y_pred = model.predict(X_val_rfe)
    r2 = r2_score(y_val, y_pred)
    print(f"{name} - Validation R² Score: {r2:.4f}")

    if r2 > best_score:
        best_score = r2
        best_model = model

print(f"\nBest Model: {best_model.__class__.__name__} with R² = {best_score:.4f}")

# Step 6: Evaluate Final Model on Test Set
y_test_pred = best_model.predict(X_test_rfe)
final_r2 = r2_score(y_test, y_test_pred)
print(f"\nFinal Test R² Score: {final_r2:.4f}")

