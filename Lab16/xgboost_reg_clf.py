import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBRegressor, XGBClassifier

def load_regression_data():
    data = pd.read_csv("Boston.csv")
    X = data.drop(columns=["Unnamed: 0", "medv"]).values
    y = data["medv"].values
    return train_test_split(X, y, test_size=0.2, random_state=42)

def load_classification_data():
    data = pd.read_csv("Weekly.csv")
    X = data.drop(columns=["Direction"]).values
    y = LabelEncoder().fit_transform(data["Direction"].values)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def preprocess(X_train, X_test):
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)

def run_regression_models():
    X_train, X_test, y_train, y_test = load_regression_data()
    X_train, X_test = preprocess(X_train, X_test)

    print("\n--- Gradient Boosting Regressor ---")
    gbr = GradientBoostingRegressor(random_state=42)
    gbr.fit(X_train, y_train)
    preds_gbr = gbr.predict(X_test)
    print(f"R² Score: {r2_score(y_test, preds_gbr):.4f}")

    print("\n--- XGBoost Regressor ---")
    xgbr = XGBRegressor(random_state=42)
    xgbr.fit(X_train, y_train)
    preds_xgb = xgbr.predict(X_test)
    print(f"R² Score: {r2_score(y_test, preds_xgb):.4f}")

def run_classification_models():
    X_train, X_test, y_train, y_test = load_classification_data()
    X_train, X_test = preprocess(X_train, X_test)

    print("\n--- Gradient Boosting Classifier ---")
    gbc = GradientBoostingClassifier(random_state=42)
    gbc.fit(X_train, y_train)
    preds_gbc = gbc.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds_gbc))
    print(classification_report(y_test, preds_gbc))

    print("\n--- XGBoost Classifier ---")
    xgbc = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgbc.fit(X_train, y_train)
    preds_xgb = xgbc.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds_xgb))
    print(classification_report(y_test, preds_xgb))

def main():
    run_regression_models()
    run_classification_models()

if __name__ == "__main__":
    main()
