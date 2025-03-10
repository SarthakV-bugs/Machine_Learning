import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Load the dataset from a CSV file
data = pd.read_csv("new_breast_cancer.csv")
headers = ['Age', 'Menopause', 'Tumor_Size', 'Lymph_Nodes', 'Hormone_Therapy', 'Tumor_Grade', 'Breast', 'Quadrant', 'Irradiation', 'Recurrence_Status']

# Add headers
data.columns = headers

# Check for NaN values in the dataset
if data.isna().any().any():
    print("Warning: Dataset contains NaN values!")
    print(data.isna().sum())  # Print the count of NaN values per column
else:
    print("No NaN values in the dataset.")

# Select input features (X) and target variable (y)
X = data.iloc[:, :-1].astype(str)
y = data.iloc[:, -1].astype(str)

# Split dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# One-Hot Encoding for categorical features
onehot_encoder = OneHotEncoder(sparse_output=False)

# Fit the encoder only on the training data and transform the training data
X_train_encoded = onehot_encoder.fit_transform(X_train)

# Transform the test data using the already fitted encoder (without fitting it again)
X_test_encoded = onehot_encoder.transform(X_test)

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
