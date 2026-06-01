"""Implementing RBF kernel and comparing it with Polynomial kernel"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import svm

data = {
    "x1":[6,6,8,8,8,9,9,10,10,11,11],
    "x2":[5,9,6,8,10,2,5,10,13,5,8],
    "label":["blue","blue","red","red","red","blue","red","red","blue","red","red"]
}

df = pd.DataFrame(data)

#extract x and y
x = df[["x1","x2"]].to_numpy()
y = df["label"].to_numpy()

#label encoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)


#Train SVM with RBF kernel
"""
gamma controls how far the influence of a single training example reaches
"""
clf_rbf = svm.SVC(kernel='rbf', gamma=0.5, C=1.0)
clf_rbf.fit(x,y_encoded)

#Train SVM with Polynomial kernel
clf_poly = svm.SVC(kernel='poly', degree=3, C=1.0)
clf_poly.fit(x,y_encoded)


###Support vectors
print(clf_rbf.support_vectors_)
print(clf_poly.support_vectors_)
print(clf_poly.support_)
print(clf_rbf.support_)
#INSIGHTS
"""Polynomial	4	Boundary fits data with less complexity
RBF	8	Needs more vectors to draw a curvy boundary"""


#LOOCV technique
"""As the dataset is very small, we avoid test train split and instead choose LOOCV method"""
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score

loo = LeaveOneOut()

score_rbf = cross_val_score(clf_rbf, x, y_encoded)
score_poly = cross_val_score(clf_poly,x, y_encoded)

print("LOOCV Accuracy (RBF):", score_rbf.mean())
print("LOOCV Accuracy (Polynomial):", score_poly.mean())


from sklearn.inspection import DecisionBoundaryDisplay
#decision boundaries
def plot_decision_boundaries(kernel, ax, title, sv_label_tag):
#train clf

    clf = svm.SVC(kernel=kernel, C=1.0,gamma=0.5 if kernel == 'rbf' else 'scale',
    degree = 3 if kernel == 'poly' else 3)
    clf.fit(x,y_encoded)

    #set up plot region
    x_min, x_max= x[:,0].min() - 2, x[:,0].max() + 2
    y_min, y_max= x[:,1].min() - 2, x[:,1].max() + 2 #boundarise from feature space X and not from labels y
    ax.set_xlim(x_min,x_max)
    ax.set_ylim(y_min,y_max)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")


    # Plot decision regions
    DecisionBoundaryDisplay.from_estimator(
        clf, x, ax=ax, response_method="predict",
        plot_method="pcolormesh", alpha=0.3, cmap='coolwarm'
    )
    DecisionBoundaryDisplay.from_estimator(
        clf, x, ax=ax, response_method="decision_function",
        plot_method="contour", levels=[-1, 0, 1],
        colors=["k", "k", "k"], linestyles=["--", "-", "--"]
    )

    # Plot support vectors with labels
    sv = x[clf.support_]
    ax.scatter(sv[:, 0], sv[:, 1], s=150, facecolors="none", edgecolors="k", linewidths=1.5,
               label=f'Support Vectors ({sv_label_tag})')
    for i, (x1, x2) in enumerate(sv):
        ax.text(x1 + 0.3, x2 + 0.3, f'SV ({sv_label_tag})', fontsize=8, color='black')

    # Plot original data
    scatter = ax.scatter(x[:, 0], x[:, 1], c=y_encoded, cmap='coolwarm', s=40, edgecolors="k", label='Data Points')
    ax.legend(loc='upper right')
    ax.set_title(title)


 #Plotting both kernels
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
plot_decision_boundaries(kernel='poly', ax=axs[0], title='Polynomial Kernel', sv_label_tag='Poly')
plot_decision_boundaries(kernel='rbf', ax=axs[1], title='RBF Kernel', sv_label_tag='RBF')
plt.suptitle("SVM Decision Boundaries with Support Vectors Labeled by Kernel", fontsize=16, y=1.03)
plt.tight_layout()
plt.show()