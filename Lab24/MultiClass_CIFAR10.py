import os
import numpy as np
from PIL import Image

# CIFAR-10 Dataset path
data_dir = "/home/ibab/PycharmProjects/ML-Lab/cifar10img/cifar10/cifar10/train"

# Lists to store image arrays and corresponding labels
x = []
y = []

# Iterate through each class directory
for label in sorted(os.listdir(data_dir)):  # ensures alphabetical order
    class_path = os.path.join(data_dir, label)
    if not os.path.isdir(class_path):
        continue
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        try:
            # Load the image and preprocess it
            img = Image.open(img_path).convert('RGB')
            img = img.resize((32, 32))  # resize if needed
            img_array = np.array(img)

            # Append to dataset
            x.append(img_array)
            y.append(label)

        except Exception as e:
            print(f"Error reading {img_path}: {e}")

# Convert to numpy arrays
x = np.array(x)
y = np.array(y)

# Print dataset info
print("Dataset shape:", x.shape)     # should be (num_samples, 32, 32, 3)
print("Class labels:", np.unique(y))  # should list 10 CIFAR-10 class names


# Flatten and normalize
x_flat = x.reshape(len(x), -1) / 255.0
print(x_flat)

# Encode string labels to integers
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(y_encoded)


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(x_flat, y_encoded, test_size=0.2, random_state=42)

# KNN Classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Prediction and accuracy
y_pred = knn.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("KNN Accuracy:", acc)

