#weekly dataset for Gradient boosting classification
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import  GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
weekly_data = pd.read_csv("Weekly.csv")

# Display dataset info
print(weekly_data.info())  # Summary of dataset
print(weekly_data.head())  # First 5 rows
print("\nMissing Values:\n", weekly_data.isnull().sum())  # Check missing values

# Histogram for categorical target variable
plt.figure(figsize=(8,5))
sns.countplot(x="Direction", data=weekly_data, hue="Direction", legend=False)
plt.xlabel("Market Direction")
plt.ylabel("Count")
plt.title("Distribution of Market Directions")
plt.show()

# Convert categorical target to numerical (Up = 1, Down = 0)
weekly_data["Direction"] = weekly_data["Direction"].map({"Up": 1, "Down": 0})

# Compute correlation matrix
corr_matrix = weekly_data.corr()

# Heatmap for feature correlations
plt.figure(figsize=(10,6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlations")
plt.show()


#extract data and target variable
x = weekly_data.drop(columns=["Direction"])
y = weekly_data["Direction"]

#split the dataset into train and test
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)

#train the model
gd_clf_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3,random_state=42)
gd_clf_model.fit(x_train,y_train)

#prediction
y_pred = gd_clf_model.predict(x_test)



# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")

# Print classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Display confusion matrix
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues", fmt="d", xticklabels=["Down", "Up"], yticklabels=["Down", "Up"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()