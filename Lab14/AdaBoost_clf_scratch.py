
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier

iris = load_iris()
x = iris.data
y = iris.target


def preprocess(X_train, X_test, y_train, y_test):
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train_enc, y_test_enc, le


def train_adaboost_samme(X, y, num_estimators=50):
    n_samples = X.shape[0]
    labels = np.unique(y)
    K = len(labels)

    weights = np.ones(n_samples) / n_samples

    models = []
    alphas = []

    for t in range(num_estimators):
        stump = DecisionTreeClassifier(max_depth=1)
        stump.fit(X, y, sample_weight=weights)
        preds = stump.predict(X)

        wrong = (preds != y).astype(int)
        err = np.dot(weights, wrong)

        err = max(err, 1e-10)
        if err >= 1 - 1e-10:
            break

        alpha = np.log((1 - err) / err) + np.log(K - 1)

        # Update weights
        weights = weights * np.exp(alpha * wrong)
        weights = weights / np.sum(weights)

        models.append(stump)
        alphas.append(alpha)

    return models, alphas, labels

def adaboost_predict(models, alphas, labels, X):
    final_votes = np.zeros((X.shape[0], len(labels)))

    for model, alpha in zip(models, alphas):
        preds = model.predict(X)
        for i in range(len(preds)):
            final_votes[i, preds[i]] += alpha

    return labels[np.argmax(final_votes, axis=1)]


def main():
    # Load & split

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    x_train_scaled, x_test_scaled, y_train_enc, y_test_enc, encoder = preprocess(x_train, x_test, y_train, y_test)

    # scikit-learn version
    print("\n Scikit-learn AdaBoost ")
    model_sklearn = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=50)
    model_sklearn.fit(x_train_scaled, y_train_enc)
    y_pred_sklearn = model_sklearn.predict(x_test_scaled)
    print(classification_report(y_test_enc, y_pred_sklearn))

    # from scratch
    print("\n Scratch AdaBoost")
    models, alphas, labels = train_adaboost_samme(x_train_scaled, y_train_enc, num_estimators=50)
    y_pred_custom = adaboost_predict(models, alphas, labels, x_test_scaled)
    print(classification_report(y_test_enc, y_pred_custom))

if __name__ == "__main__":
    main()
