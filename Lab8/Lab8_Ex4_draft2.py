import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

# Load the dataset from a CSV file
data = pd.read_csv("Breast_cancer_2.csv")
headers = ['Age', 'Menopause', 'Tumor_Size', 'Lymph_Nodes', 'Hormone_Therapy', 'Tumor_Grade', 'Breast', 'Quadrant', 'Irradiation', 'Recurrence_Status']

# Add headers
data.columns = headers

# Replace empty strings with NaN
data = data.replace('', np.nan)

# Check for NaN values in the dataset
if data.isna().any().any():
    print("Warning: Dataset contains NaN values!")
    print(data.isna().sum())  # Print the count of NaN values per column

    # Impute missing values using the most frequent value (mode) for each column
    for col in data.columns:
        if data[col].isnull().any():
            most_frequent_value = data[col].mode()[0]  # Get the first mode if there are multiple
            data[col] = data[col].fillna(most_frequent_value)
    print("NaN values imputed with mode.")
else:
    print("No NaN values in the dataset.")

# Select input features (X) and target variable (y)
X = data.iloc[:, :-1].astype(str)
y = data.iloc[:, -1].astype(str)

# Split dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the columns for different encoding strategies
categorical_features = ['Age', 'Menopause', 'Hormone_Therapy', 'Breast', 'Quadrant', 'Irradiation']  # Include 'Age' here
ordinal_features = ['Tumor_Size', 'Lymph_Nodes', 'Tumor_Grade']  # 'Age' is removed

# Define the order of categories for ordinal features
# age_categories = ['20-29', '30-39', '40-49', '50-59', '60-69', '70-79'] # No longer needed
menopause_categories = ['lt40', 'premeno', 'ge40']
tumor_size_categories = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54']
lymph_nodes_categories = ['0-2', '3-5', '6-8', '9-11', '12-14', '15-17', '24-26']
hormone_therapy_categories = ['no', 'yes']
tumor_grade_categories = ['1', '2', '3']
quadrant_categories = ['central', 'left_up', 'left_low', 'right_up', 'right_low']

# Create transformers
onehot_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
ordinal_transformer = OrdinalEncoder(categories=[tumor_size_categories, lymph_nodes_categories, tumor_grade_categories],
                                      handle_unknown='use_encoded_value', unknown_value=-1)

# Create a column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', onehot_transformer, categorical_features),
        ('ordinal', ordinal_transformer, ordinal_features)],
    remainder='passthrough')  # This is important to handle any other columns

# Create a pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Fit and transform the training data
X_train_encoded = pipeline.fit_transform(X_train)

# Transform the test data
X_test_encoded = pipeline.transform(X_test)

# Label Encoding for target variable
label_encoder = LabelEncoder()

# Fit and transform on the training target
y_train_encoded = label_encoder.fit_transform(y_train)

# Transform the test target using the fitted encoder
y_test_encoded = label_encoder.transform(y_test)

# Print all rows of the encoded data
print("Training Input (Encoded):")
pd.set_option('display.max_rows', None)  # This allows all rows to be printed
print(pd.DataFrame(X_train_encoded))

print("Testing Input (Encoded):")
print(pd.DataFrame(X_test_encoded))

print("Training Output (Encoded):")
print(pd.DataFrame(y_train_encoded))

print("Testing Output (Encoded):")
print(pd.DataFrame(y_test_encoded))
