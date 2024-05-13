import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree


# fig, axs = plt.subplots(1, 2, sharex='col')


## ЗАДАНИЕ 1.1 - 1.3 -----------------------------------------------------------------

dFrame = pd.read_csv("./lab4_5.csv")

# print(dFrame.head())
# print(dFrame.describe())

# sns.scatterplot(dFrame, x = "X1", y = "X2", hue="Class", palette="Set1")

# plt.show()

## ЗАДАНИЕ 1.4 -----------------------------------------------------------------------

le = LabelEncoder()
le.fit(dFrame["Class"])

dFrame["Class"] = le.transform(dFrame["Class"])

## ЗАДАНИЕ 1.5 -----------------------------------------------------------------------

dFrameTrain, dFrameTest = train_test_split(dFrame, test_size=0.3, random_state=42)

# sns.scatterplot(dFrameTrain, x = "X1", y = "X2", hue="Class", palette="Set1", ax = axs[0]).set(title="Train")
# sns.scatterplot(dFrameTest, x = "X1", y = "X2", hue="Class", palette="Set1", ax = axs[1]).set(title="Test")

# print(dFrameTrain.describe())
# print(dFrameTest.describe())

# plt.show()



## ЗАДАНИЕ 2.1 -----------------------------------------------------------------------

testX = dFrameTest.iloc[:, 0:2]
trainX = dFrameTrain.iloc[:, 0:2]

testY = dFrameTest.iloc[:, 2:3]
trainY = dFrameTrain.iloc[:, 2:3]

def accuracyScore(m, weight):
    knn = KNeighborsClassifier(m, weights=weight)
    knn.fit(trainX, np.array(trainY).ravel())
    y_pred = knn.predict(testX)
    return accuracy_score(testY, y_pred)

X = [x for x in range (1,15)]
uniformScore = [accuracyScore(k, 'uniform') for k in X]
distanceScore = [accuracyScore(k, 'distance') for k in X]

# sns.relplot(pd.DataFrame(uniformScore, columns=["Score"]).assign(k=X), x="k", y="Score", kind="line").set(title="uniform")
# sns.relplot(pd.DataFrame(distanceScore, columns=["Score"]).assign(k=X), x="k", y="Score", kind="line").set(title="distance")


## ЗАДАНИЕ 2.2 -----------------------------------------------------------------------

knn = KNeighborsClassifier(9, weights="distance")
knn.fit(trainX, np.array(trainY).ravel())
y_pred = knn.predict(testX)


def display(classificator):
    display = DecisionBoundaryDisplay.from_estimator(classificator, testX, xlabel="X1", ylabel="X2", response_method="predict", alpha=0.7, grid_resolution=200)
    display.ax_.scatter(testX["X1"], testX["X2"], c=y_pred, edgecolors="k")
    plt.show()


## ЗАДАНИЕ 2.3 -----------------------------------------------------------------------

def confMatrix(classes):
    confMatrix = confusion_matrix(testY, y_pred)
    confDisplay = ConfusionMatrixDisplay(confusion_matrix=confMatrix, display_labels=classes)
    confDisplay.plot()
    plt.show()

# confMatrix(knn.classes_)

## ЗАДАНИЕ 2.4 -----------------------------------------------------------------------

def scores():
    precisionScore = precision_score(testY, y_pred, average='binary') 
    recallScore = recall_score(testY, y_pred, average='binary') 
    f1Score = f1_score(testY, y_pred, average='binary') 

    print("precision score: ", precisionScore)
    print("recall score: ", recallScore)
    print("f1 score: ", f1Score)

# scores()

## ЗАДАНИЕ 2.5 ----------------------l-------------------------------------------------

def ROC_AUC(title_):
    knnAuc = roc_auc_score(y_pred, testY)
    knnFpr, knnTpr, knnT = roc_curve(testY, y_pred)
    print(title_ + " AUC: ", knnAuc)
    sns.relplot(pd.DataFrame(knnFpr, columns=["fpr"]).assign(tpr=knnTpr), x="fpr", y="tpr", kind="line").set(title=title_ + ' ROC')
    plt.show()

# ROC_AUC("KNN ROC")
    
## ЗАДАНИЕ 3.1 -----------------------------------------------------------------------

logreg = LogisticRegression(penalty = None)
logreg.fit(trainX, np.array(trainY).ravel())
y_pred = logreg.predict(testX)

# names = ['l2', 'l1', 'None']
# scores = [0.8166666666666667, 0.8333333333333334, 0.8333333333333334]

# plt.bar(names, scores)
# plt.show()



## ЗАДАНИЕ 3.2 -----------------------------------------------------------------------

# display(logreg)



## ЗАДАНИЕ 3.3 -----------------------------------------------------------------------

# confMatrix(logreg.classes_)



## ЗАДАНИЕ 3.3 -----------------------------------------------------------------------

# scores()



## ЗАДАНИЕ 3.4 -----------------------------------------------------------------------

# ROC_AUC("LogReg")



## ЗАДАНИЕ 4.1 -----------------------------------------------------------------------


SVM = SVC(kernel='linear')
SVM.fit(trainX, np.array(trainY).ravel())
y_pred = SVM.predict(testX)

# X = [x for x in range (1,10)]
# polyScore = [SVMa(k) for k in X]

# sns.relplot(pd.DataFrame(polyScore, columns=["Score"]).assign(k=X), x="k", y="Score", kind="line").set(title="poly")
# plt.show()

# names = ['linear', 'rbf', 'poly']
# scores = [0.8166666666666667, 0.8, 0.8]

# plt.bar(names, scores)
# plt.show()



## ЗАДАНИЕ 4.2 -----------------------------------------------------------------------

# display(SVM)



## ЗАДАНИЕ 4.3 -----------------------------------------------------------------------

# confMatrix(SVM.classes_)



## ЗАДАНИЕ 4.3 -----------------------------------------------------------------------

# scores()



## ЗАДАНИЕ 4.4 -----------------------------------------------------------------------

# ROC_AUC("SVM")



## ЗАДАНИЕ 5.1 -----------------------------------------------------------------------

dtc = DecisionTreeClassifier(criterion = 'entropy', max_depth=1, max_leaf_nodes=6)
dtc.fit(trainX,trainY)
y_pred = dtc.predict(testX)


## ЗАДАНИЕ 5.2 -----------------------------------------------------------------------

# display(SVM)



## ЗАДАНИЕ 5.3 -----------------------------------------------------------------------

# confMatrix(SVM.classes_)



## ЗАДАНИЕ 5.4 -----------------------------------------------------------------------

# scores()



## ЗАДАНИЕ 5.5 -----------------------------------------------------------------------

# ROC_AUC("Decision Tree")



## ЗАДАНИЕ 5.6 -----------------------------------------------------------------------

# plot_tree(dtc, filled = True)

# plt.show()



