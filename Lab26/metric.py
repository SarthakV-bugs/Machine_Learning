#
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_curve, \
    roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
data = pd.read_csv("heart.csv")
print(data.keys())
#extract x and y

x = data.drop(columns=["output"])
y = data["output"]

print(x)
print(y)

#split the data
x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.2,random_state=42)

#train logistic model
model = LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)

#get the probabilities
print(model.classes_) #[0,1]
y_probs = model.predict_proba(x_test)[:,1] #probability of class 1
# print(y_probs)

#define thresholds to vary
thresholds = [0.3,0.4,0.5,0.6,0.7]

#calculate metrics for each thresholds
results = []
for thresh in thresholds:
    #make pred based on each threshold
    y_pred = (y_probs >= thresh).astype(int)
    print(y_pred)

    #calculate the confusion matrix for TP,FP,TN,FN
    tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred)  # same as recall
    specificity = tn / (tn + fp)
    f1 = f1_score(y_test, y_pred)

    # Store
    # results
    results.append({
        'Threshold': thresh,
        'TP': tp,
        'FP': fp,
        'TN': tn,
        'FN': fn,
        'Accuracy': accuracy,
        'Precision': precision,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'F1-score': f1
    })

print(results)

#results into dataframe
results_df = pd.DataFrame(results)
print(results_df)



# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_probs)
auc_score = roc_auc_score(y_test, y_probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Heart Disease Prediction')
plt.legend()
plt.show()

print(f"\nAUC Score: {auc_score:.4f}")