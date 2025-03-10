import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split


#use case of kfold in sklearn

# X = np.array([[1,2],[3,4],[5,6]])
# Y = np.array([3,4,5])
# kf = KFold(n_splits=3)
# kf.get_n_splits(X)
# print(kf)
#
# for i, (train_index, test_index) in enumerate(kf.split(X)):
#     print(f"Fold {i}:")
#     print(f" Train:  index={train_index}")
#     print(f" Test:   index={test_index}")


def load_data(filepath, target_column, columns_to_drop):
    df = pd.read_csv(filepath)
    x = df.drop(columns=columns_to_drop).values
    y = df[target_column].values

    x = np.c_[np.ones(x.shape[0]), x] #to add the intercept term
    return x, y

def k_foldsplit(x,k):
    # first obtain the indices of all the samples

    indices = np.arange(len(x))
    # print(indices)

    #distribute the indices randomly into k folds
    np.random.shuffle(indices)

    #get the fold size
    fold_size =  len(x)//k

    #create an empty list to store the folds
    folds = []
    for i in range(k):
        start = i * fold_size #starting index
        end = (i+1) * fold_size  #ending index for the indices to be picked
        val_indices = indices[start:end]
        train_indices = np.setdiff1d(indices, val_indices)  #setdiff1d returns the unique elements in first array when compared to the second array
        folds.append((train_indices, val_indices))
    return folds

def cross_validate(x_train, y_train, k=10, alpha=0.01, iterations=1000):

    folds = k_foldsplit(x_train,k)
    r2_score = [] #store the r2 score for each fold to compute the average

    for x_train_id, val_id in folds:
        x_train, x_val = x[x_train_id], x[val_id]
        y_



if __name__ == "__main__":
    x, y = load_data("/Lab4/simulated_data_multiple_linear_regression_for_ML.csv",
                     "disease_score_fluct",
                     ["disease_score", "disease_score_fluct"])

    x_train,x_test, y_train,y_test = train_test_split(x,y,test_size=0.30,random_state=42)
    k=10
    k_foldsplit(x_train,k)
