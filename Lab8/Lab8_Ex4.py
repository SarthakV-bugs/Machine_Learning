import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, OneHotEncoder

# Load the dataset from a CSV file
data = pd.read_csv("/Lab8/new_breast_cancer.csv")
headers = ['Age', 'Menopause', 'Tumor_Size', 'Lymph_Nodes', 'Hormone_Therapy', 'Tumor_Grade', 'Breast', 'Quadrant', 'Irradiation', 'Recurrence_Status']

# Add headers
data.columns = headers

data.to_csv('Breast_cancer_2.csv', index=False)

# Select input features (X) and target variable (y)
X = data.iloc[:, :-1].astype(str)
y = data.iloc[:, -1].astype(str)

#one hot encoder

onehot_encoder = OneHotEncoder(sparse_output=False)
X = onehot_encoder.fit_transform(X)
#.fit - learns all the unique values in a column
#transform converts the categorical data into a one hot encoded form.

#ordinal encode the target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

print('Input', X.shape)
print(X[:5, :])



#
# # Initialize the OrdinalEncoder
# ordinal_encoder = OrdinalEncoder()
#
# # Perform ordinal encoding on X
# X_encoded = ordinal_encoder.fit_transform(X)
#
# # Perform label encoding on the target variable y
# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)
#
# # Print the transformed data
# print('Input', X_encoded.shape)
# print(X_encoded[:5, :])
# print('Output', y_encoded.shape)
# print(y_encoded[:5])
#
# # Optionally, find the corresponding categories for each feature after encoding
# ordinal_categories = ordinal_encoder.categories_
# for i, categories in enumerate(ordinal_categories):
#     print(f"Feature {X.columns[i]}: {categories}")
