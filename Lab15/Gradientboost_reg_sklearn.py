# Implement Gradient Boost Regression and Classification using scikit-learn. Use the
# Boston housing dataset from the ISLP package for the regression problem and weekly
# dataset from the ISLP package and use Direction as the target variable for the
# classification.
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


#load datasets

housing_data = pd.read_csv("Boston.csv")

#EDA
# print(housing_data.columns)
# print(housing_data.head(5))
# print(housing_data.isnull().sum())  # Shows count of missing values per column
#
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # Histogram of the target variable
# plt.figure(figsize=(8,5))
# sns.histplot(housing_data["medv"], bins=30, kde=True)  # "medv" is the target variable
# plt.xlabel("Median House Price ($1000s)")
# plt.title("Distribution of House Prices")
# plt.show()
#
# #feature study
# corr_matrix = housing_data.corr()
# plt.figure(figsize=(10,6))
# sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
# plt.title("Feature Correlations")
# plt.show()



#Drop the column unnamed for extracting the data
x = housing_data.drop(columns=["Unnamed: 0","medv"])
#medianvalue is the target variable
y = housing_data["medv"]

print(x,y)

#split the dataset into train and test
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)

#initialize the model
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3,random_state=42)
gb_model.fit(x_train,y_train)

#prediction
y_pred = gb_model.predict(x_test)

#r2 score

r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.2f}")


# Get feature importance
feature_importance = gb_model.feature_importances_
features = x.columns

# Plot importance
plt.figure(figsize=(10,5))
plt.barh(features, feature_importance, color="skyblue")
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importance in Gradient Boosting Model")
plt.show()
