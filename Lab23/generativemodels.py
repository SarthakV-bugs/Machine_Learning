import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Load the dataset
iris = pd.read_csv("Iris.csv")

#Extract the first two features and target
x = iris[["SepalLengthCm", "SepalWidthCm"]]
y = iris["Species"]

#Add random noise to features
np.random.seed(42)
noise = np.random.normal(loc=0.0, scale=0.2, size=x.shape)
x_noisy = x + noise

# #Visualize histograms with bins
# for i in range(x_noisy.shape[1]):
#     plt.hist(x_noisy.iloc[:, i], bins=4, edgecolor='black')
#     plt.title(f"Histogram of {x_noisy.columns[i]} with 4 bins")
#     plt.xlabel(x_noisy.columns[i])
#     plt.ylabel("Frequency")
#     plt.grid(True)
#     plt.show()

#Discretization function
def equal_width_discretize(data, n_bins):
    data = np.array(data)
    min_val = np.min(data)
    max_val = np.max(data)
    bin_edges = np.linspace(min_val, max_val, n_bins + 1)
    bins = np.digitize(data, bins=bin_edges[1:-1], right=False)
    return bins

# f) Apply discretization
n_bins = 4
disc_x1 = equal_width_discretize(x_noisy.iloc[:, 0], n_bins)
disc_x2 = equal_width_discretize(x_noisy.iloc[:, 1], n_bins)

X_discrete = pd.DataFrame({
    "SepalLengthBin": disc_x1,
    "SepalWidthBin": disc_x2
})

print(X_discrete)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_discrete, y, test_size=0.3, random_state=42)
# print(X_train)

# Joint Probability Model
joint_prob_table = {}
for i in range(len(X_train)): #iterates each row
    key = (X_train.iloc[i, 0], X_train.iloc[i, 1])
    print(key)
    label = y_train.iloc[i]
    # print(label)
    if key not in joint_prob_table:
        joint_prob_table[key] = {}
    if label not in joint_prob_table[key]:
        joint_prob_table[key][label] = 0
    joint_prob_table[key][label] += 1

# Normalize to get probabilities
for key in joint_prob_table:
    total = sum(joint_prob_table[key].values())
    for label in joint_prob_table[key]:
        joint_prob_table[key][label] /= total


def predict_joint_prob(x_row):
    key = (x_row.iloc[0], x_row.iloc[1])
    if key in joint_prob_table:
        return max(joint_prob_table[key], key=joint_prob_table[key].get)
    else:
        return y_train.value_counts().idxmax()


y_pred_joint = X_test.apply(predict_joint_prob, axis=1)

# Accuracy
joint_accuracy = accuracy_score(y_test, y_pred_joint)
print(f"Joint Probability Model Accuracy: {joint_accuracy:.2f}")

#Decision Tree Classifier with max_depth = 2
tree = DecisionTreeClassifier(max_depth=2, random_state=42)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)

tree_accuracy = accuracy_score(y_test, y_pred_tree)
print(f"Decision Tree Accuracy (max_depth=2): {tree_accuracy:.2f}")
