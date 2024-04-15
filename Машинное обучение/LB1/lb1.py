import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


dFrame = pd.read_csv("iris.csv", index_col=0)

print(dFrame.head())
print(dFrame.describe())

def changeLabel(label):
    if label == 0:
        return "Iris-setosa"
    if label == 1:
        return "Iris-versicolor"
    if label == 2:
        return "Iris-virginica"

dFrame["target"] = dFrame["target"].apply(changeLabel)

# dFrame.to_csv("iris_change_label.csv", index=False)

sns.set_theme(style="white")
# g = sns.PairGrid(dFrame, diag_sharey=False, hue="target", palette="Set1")

# g.map_upper(sns.scatterplot, s=15)
# g.map_lower(sns.kdeplot)
# g.map_diag(sns.kdeplot, lw=2)
# g.add_legend()

# plt.show()

# fig, axs = plt.subplots(1, 4, sharex='col')
# columnsCount = 15

# sns.histplot(dFrame, ax = axs[0], bins=columnsCount, x = "sepal length (cm)") 
# sns.histplot(dFrame, ax = axs[1], bins=columnsCount,x = "sepal width (cm)")
# sns.histplot(dFrame, ax = axs[2], bins=columnsCount, x = "petal length (cm)")
# sns.histplot(dFrame, ax = axs[3], bins=columnsCount, x = "petal width (cm)")

# plt.show()

# # Пересечение
# sns.histplot(dFrame, ax = axs[0], bins=columnsCount, x = "sepal length (cm)", hue="target", palette="Set1")
# sns.histplot(dFrame, ax = axs[1], bins=columnsCount,x = "sepal width (cm)", hue="target", palette="Set1")
# sns.histplot(dFrame, ax = axs[2], bins=columnsCount, x = "petal length (cm)", hue="target", palette="Set1")
# sns.histplot(dFrame, ax = axs[3], bins=columnsCount, x = "petal width (cm)", hue="target", palette="Set1")

# # Накладывание
# sns.histplot(dFrame, ax = axs[0], bins=columnsCount, x = "sepal length (cm)", hue="target", palette="Set1", multiple="stack")
# sns.histplot(dFrame, ax = axs[1], bins=columnsCount,x = "sepal width (cm)", hue="target", palette="Set1", multiple="stack")
# sns.histplot(dFrame, ax = axs[2], bins=columnsCount, x = "petal length (cm)", hue="target", palette="Set1", multiple="stack")
# sns.histplot(dFrame, ax = axs[3], bins=columnsCount, x = "petal width (cm)", hue="target", palette="Set1", multiple="stack")

##Ступеньки
# sns.histplot(dFrame, ax = axs[0], bins=columnsCount, x = "sepal length (cm)", hue="target", palette="Set1", element="step")
# sns.histplot(dFrame, ax = axs[1], bins=columnsCount,x = "sepal width (cm)", hue="target", palette="Set1", element="step")
# sns.histplot(dFrame, ax = axs[2], bins=columnsCount, x = "petal length (cm)", hue="target", palette="Set1", element="step")
# sns.histplot(dFrame, ax = axs[3], bins=columnsCount, x = "petal width (cm)", hue="target", palette="Set1", element="step")

# ##Ядерная оценка
# sns.histplot(dFrame, ax = axs[0], bins=columnsCount, x = "sepal length (cm)", hue="target", palette="Set1", element="step", kde=True)
# sns.histplot(dFrame, ax = axs[1], bins=columnsCount,x = "sepal width (cm)", hue="target", palette="Set1", element="step", kde=True)
# sns.histplot(dFrame, ax = axs[2], bins=columnsCount, x = "petal length (cm)", hue="target", palette="Set1", element="step", kde=True)
# sns.histplot(dFrame, ax = axs[3], bins=columnsCount, x = "petal width (cm)", hue="target", palette="Set1", element="step", kde=True)

# plt.show()

# # ЗАДАНИЕ 2 

# dataArray = np.genfromtxt("iris.csv", delimiter=",")[1:]

# print(dataArray[:10])

# rowsCount, columnsCount = np.shape(dataArray) 
# mean = np.mean(dataArray, axis=0) 
# std = np.std(dataArray, axis=0)
# min = np.min(dataArray, axis=0)
# qQuarter = np.quantile(dataArray, 0.25, axis=0)
# qHalf = np.quantile(dataArray, 0.5, axis=0)
# qThreeQuarter = np.quantile(dataArray, 0.75, axis = 0)
# max = np.max(dataArray, axis=0)

# print(["sepal length (cm)", "sepal width (cm)",  "petal length (cm)",  "petal width (cm)"])
# print("count: ", (str(rowsCount) + " ")*4 )
# print("mean: ", mean[1:5])
# print("std: ", std[1:5])
# print("min: ", min[1:5])
# print("25%: ", qQuarter[1:5])
# print("50%: ", qHalf[1:5])
# print("75%: ", qThreeQuarter[1:5])
# print("max: ", max[1:5])


# # ЗАДАНИЕ 3 - ТОЖЕ САМОЕ, ЧТО И В ПЕРВОМ 


# # ЗАДАНИЕ 4

# target = dFrame["target"] 

# le = LabelEncoder()
# le.fit(target)

# print(le.classes_)
# print(le.transform(le.classes_))


# ohe = OneHotEncoder(sparse_output=False)
# ohe.fit(pd.DataFrame(target))

# print(ohe.categories_[0])
# print(ohe.transform(pd.DataFrame(ohe.categories_[0])))



# signs = np.array(dFrame.drop("target", axis=1))

# stdS = StandardScaler()
# stdS.fit(signs) 
# stdRes = stdS.transform(signs)

# fig, axs = plt.subplots(1, 4, sharex='col')
# columnsCount = 15

# sns.histplot(stdRes[:,0], ax = axs[0], bins=columnsCount)
# sns.histplot(stdRes[:,1], ax = axs[1], bins=columnsCount)
# sns.histplot(stdRes[:,2], ax = axs[2], bins=columnsCount)
# sns.histplot(stdRes[:,3], ax = axs[3], bins=columnsCount)

# plt.show()

# minMaxS = MinMaxScaler()
# minMaxS.fit(signs) 
# minMaxSRes = minMaxS.transform(signs)

# sns.histplot(minMaxSRes[:,0], ax = axs[0], bins=columnsCount)
# sns.histplot(minMaxSRes[:,1], ax = axs[1], bins=columnsCount)
# sns.histplot(minMaxSRes[:,2], ax = axs[2], bins=columnsCount)
# sns.histplot(minMaxSRes[:,3], ax = axs[3], bins=columnsCount)

# plt.show()

# MaxAbsS = MaxAbsScaler()
# MaxAbsS.fit(signs) 
# MaxAbsSRes = MaxAbsS.transform(signs)

# sns.histplot(MaxAbsSRes[:,0], ax = axs[0], bins=columnsCount)
# sns.histplot(MaxAbsSRes[:,1], ax = axs[1], bins=columnsCount)
# sns.histplot(MaxAbsSRes[:,2], ax = axs[2], bins=columnsCount)
# sns.histplot(MaxAbsSRes[:,3], ax = axs[3], bins=columnsCount)

# plt.show()

# RobS = RobustScaler()
# RobS.fit(signs) 
# RobSRes = RobS.transform(signs)

# sns.histplot(RobSRes[:,0], ax = axs[0], bins=columnsCount)
# sns.histplot(RobSRes[:,1], ax = axs[1], bins=columnsCount)
# sns.histplot(RobSRes[:,2], ax = axs[2], bins=columnsCount)
# sns.histplot(RobSRes[:,3], ax = axs[3], bins=columnsCount)

# plt.show()


# def npMinMaxScaler(Signs):
#     min = np.min(Signs, axis=0)
#     max = np.max(Signs, axis=0)
#     return((Signs-min)/(max-min))

# userMinMaxRes = npMinMaxScaler(signs)

# sns.histplot(userMinMaxRes[:,0], ax = axs[0], bins=columnsCount)
# sns.histplot(userMinMaxRes[:,1], ax = axs[1], bins=columnsCount)
# sns.histplot(userMinMaxRes[:,2], ax = axs[2], bins=columnsCount)
# sns.histplot(userMinMaxRes[:,3], ax = axs[3], bins=columnsCount)

# plt.show()

# print(pd.DataFrame(minMaxSRes).describe())
# print(pd.DataFrame(userMinMaxRes).describe())




newDFrame = dFrame.loc[(np.quantile(dFrame["sepal width (cm)"], 0.25, axis=0) <= dFrame["sepal width (cm)"]) & (np.quantile(dFrame["sepal width (cm)"], 0.75, axis=0) >= dFrame["sepal width (cm)"])]
newDFrame = dFrame.iloc[:,[0,2,4]] 
newDFrame = newDFrame[newDFrame['target'] != 'Iris-setosa']

print(newDFrame)



# # ЗАДАНИЕ 5


pca = PCA(n_components=2)
xPca = pca.fit_transform(dFrame)
sns.scatterplot(x=xPca[:, 0], y=xPca[:, 1], hue=dFrame['target'] , palette='Set1')

plt.show()


tsne = TSNE(n_components=2, random_state=42)
xTsne = tsne.fit_transform(dFrame)
sns.scatterplot(x=xTsne[:, 0], y=xTsne[:, 1], hue=dFrame['target'] , palette='Set1')

plt.show()



# # ЗАДАНИЕ 3
dFrame = pd.read_csv("lab1_var1.csv", index_col=0)

print(dFrame.head())

print(dFrame.describe())

def changeLabel(label):
    if label == 0:
        return "Iris-setosa"
    if label == 1:
        return "Iris-versicolor"
    if label == 2:
        return "Iris-virginica"

dFrame["label"] = dFrame["label"].apply(changeLabel)

dFrame.to_csv("lab1_var1_change_label.csv", index=False)

# sns.set_theme(style="white")
# g = sns.PairGrid(dFrame, diag_sharey=False, hue="label", palette="Set1")

# g.map_upper(sns.scatterplot, s=15)
# g.map_lower(sns.kdeplot)
# g.map_diag(sns.kdeplot, lw=2)
# g.add_legend()

# plt.show()

# fig, axs = plt.subplots(1, 4, sharex='col')
# columnsCount = 30

# sns.histplot(dFrame, ax = axs[0], bins=columnsCount, x = "first")
# sns.histplot(dFrame, ax = axs[1], bins=columnsCount,x = "second")
# sns.histplot(dFrame, ax = axs[2], bins=columnsCount, x = "third")
# sns.histplot(dFrame, ax = axs[3], bins=columnsCount, x = "fourth")

# plt.show()

# # Пересечение
# sns.histplot(dFrame, ax = axs[0], bins=columnsCount, x = "first", hue="label", palette="Set1")
# sns.histplot(dFrame, ax = axs[1], bins=columnsCount,x = "second", hue="label", palette="Set1")
# sns.histplot(dFrame, ax = axs[2], bins=columnsCount, x = "third", hue="label", palette="Set1")
# sns.histplot(dFrame, ax = axs[3], bins=columnsCount, x = "fourth", hue="label", palette="Set1")

# plt.show()

# # Накладывание
# sns.histplot(dFrame, ax = axs[0], bins=columnsCount, x = "first", hue="label", palette="Set1", multiple="stack")
# sns.histplot(dFrame, ax = axs[1], bins=columnsCount,x = "second", hue="label", palette="Set1", multiple="stack")
# sns.histplot(dFrame, ax = axs[2], bins=columnsCount, x = "third", hue="label", palette="Set1", multiple="stack")
# sns.histplot(dFrame, ax = axs[3], bins=columnsCount, x = "fourth", hue="label", palette="Set1", multiple="stack")

# plt.show()

# ##Ступеньки 
# sns.histplot(dFrame, ax = axs[0], bins=columnsCount, x = "first", hue="label", palette="Set1", multiple="dodge", element="step")
# sns.histplot(dFrame, ax = axs[1], bins=columnsCount,x = "second", hue="label", palette="Set1", multiple="dodge", element="step")
# sns.histplot(dFrame, ax = axs[2], bins=columnsCount, x = "third", hue="label", palette="Set1", multiple="dodge", element="step")
# sns.histplot(dFrame, ax = axs[3], bins=columnsCount, x = "fourth", hue="label", palette="Set1", multiple="dodge", element="step")

# plt.show()

# ##Ядерная оценка
# sns.histplot(dFrame, ax = axs[0], bins=columnsCount, x = "first", hue="label", palette="Set1", element="step", kde=True)
# sns.histplot(dFrame, ax = axs[1], bins=columnsCount,x = "second", hue="label", palette="Set1", element="step", kde=True)
# sns.histplot(dFrame, ax = axs[2], bins=columnsCount, x = "third", hue="label", palette="Set1", element="step", kde=True)
# sns.histplot(dFrame, ax = axs[3], bins=columnsCount, x = "fourth", hue="label", palette="Set1", element="step", kde=True)

# plt.show()
