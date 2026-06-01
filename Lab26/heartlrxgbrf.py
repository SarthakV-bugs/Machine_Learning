import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def process_data(data):
    x = data.drop(columns=["output"])
    y = data['output']

    numeric_features = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak']
    categorical_features = ['cp', 'slp', 'thall']

    # One-hot encode categorical variables
    x = pd.get_dummies(x, columns=categorical_features, drop_first=True, dtype=float)

    # Scale numeric features
    scaler = StandardScaler()
    x_scaled = x.copy()
    x_scaled[numeric_features] = scaler.fit_transform(x[numeric_features])

    return x, x_scaled, y


def model(x, x_scaled, y):
    # Logistic Regression uses scaled data
    x_train_lr, x_test_lr, y_train_lr, y_test_lr = train_test_split(x_scaled, y, test_size=0.2, random_state=42)
    clf_lr = LogisticRegression(max_iter=1000, solver='liblinear')
    clf_lr.fit(x_train_lr, y_train_lr)
    y_pred_lr = clf_lr.predict(x_test_lr)
    print("Logistic Regression Accuracy:", accuracy_score(y_test_lr, y_pred_lr))

    # Random Forest uses unscaled data
    x_train_rf, x_test_rf, y_train_rf, y_test_rf = train_test_split(x, y, test_size=0.2, random_state=42)
    clf_rf = RandomForestClassifier()
    clf_rf.fit(x_train_rf, y_train_rf)
    y_pred_rf = clf_rf.predict(x_test_rf)
    print("Random Forest Accuracy:", accuracy_score(y_test_rf, y_pred_rf))

    # XGBoost uses unscaled data
    clf_xgb = XGBClassifier( eval_metric='logloss')
    clf_xgb.fit(x_train_rf, y_train_rf)
    y_pred_xgb = clf_xgb.predict(x_test_rf)
    print("XGBoost Accuracy:", accuracy_score(y_test_rf, y_pred_xgb))

    return clf_lr, clf_rf, clf_xgb


def cross_val_models(x, x_scaled, y):
    k = 10
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)

    # Logistic Regression needs scaled data
    model_lr = LogisticRegression(max_iter=1000, solver='liblinear')
    scores_lr = cross_val_score(model_lr, x_scaled, y, cv=kfold, scoring='accuracy')
    print("\n10-Fold Cross-Validation Accuracy for Logistic Regression:")
    for i, score in enumerate(scores_lr, 1):
        print(f" Fold {i}: {score:.4f}")
    print(f" Mean Accuracy: {scores_lr.mean():.4f}")

    # Random Forest uses unscaled data
    model_rf = RandomForestClassifier()
    scores_rf = cross_val_score(model_rf, x, y, cv=kfold, scoring='accuracy')
    print("\n10-Fold Cross-Validation Accuracy for Random Forest:")
    for i, score in enumerate(scores_rf, 1):
        print(f" Fold {i}: {score:.4f}")
    print(f" Mean Accuracy: {scores_rf.mean():.4f}")

    # XGBoost uses unscaled data
    model_xgb = XGBClassifier( eval_metric='logloss')
    scores_xgb = cross_val_score(model_xgb, x, y, cv=kfold, scoring='accuracy')
    print("\n10-Fold Cross-Validation Accuracy for XGBoost:")
    for i, score in enumerate(scores_xgb, 1):
        print(f" Fold {i}: {score:.4f}")
    print(f" Mean Accuracy: {scores_xgb.mean():.4f}")


def main():
    data = pd.read_csv("../ML_PRACTICE/heart.csv")

    x, x_scaled, y = process_data(data)

    # Train models and print accuracy on test split
    clf_lr, clf_rf, clf_xgb = model(x, x_scaled, y)

    # Perform 10-fold cross-validation
    cross_val_models(x, x_scaled, y)


if __name__ == '__main__':
    main()
