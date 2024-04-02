import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dFrame = pd.read_csv("E:\LETI2024\Машинное обучение\LB1\iris.csv", index_col=0)

print(dFrame.head())

# print(dFrame.describe())

# def changeLabel(label):
#     if label == 0:
#         return "Iris-setosa"
#     if label == 1:
#         return "Iris-versicolor"
#     if label == 2:
#         return "Iris-virginica"

# dFrame["target"] = dFrame["target"].apply(changeLabel)

# dFrame.to_csv("iris_change_label.csv", index=False)

sns.set_theme(style="white")
g = sns.PairGrid(dFrame, diag_sharey=False, hue="target", palette="Set1")

g.map_upper(sns.scatterplot, s=15)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot, lw=2)
g.add_legend()

plt.show()

