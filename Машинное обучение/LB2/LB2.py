import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Вариант 134Б 

# Задание 1 ---------------------------------------------------------

dFrameBlobs = pd.read_csv("LB2\data\lab2_blobs.csv")
dFrameChecker = pd.read_csv("LB2\data\lab2_checker.csv")
dFrameMoons = pd.read_csv("LB2\data\lab2_noisymoons.csv")


# print(dFrameBlobs.head())
# print(dFrameChecker.head())
# print(dFrameMoons.head())

fig, axs = plt.subplots(1, 3, sharex='col')

# sns.scatterplot(dFrameBlobs, ax = axs[0], x = "# x", y = "y").set(title="Blobs")
# sns.scatterplot(dFrameChecker, ax = axs[1], x = "# x", y = "y").set(title="Checker")
# sns.scatterplot(dFrameMoons, ax = axs[2], x = "# x", y = "y").set(title="Moons") 

# scaler = MinMaxScaler()
scaler = StandardScaler()


scaler.fit(dFrameBlobs) 
dFrameBlobsScale = scaler.transform(dFrameBlobs)

scaler.fit(dFrameChecker) 
dFrameCheckerScale = scaler.transform(dFrameChecker)

scaler.fit(dFrameMoons) 
dFrameMoonsScale = scaler.transform(dFrameMoons)

# sns.scatterplot(pd.DataFrame(dFrameBlobsScale, columns=["# x", "y"]), ax = axs[0], x = "# x", y = "y").set(title="Blobs")
# sns.scatterplot(pd.DataFrame(dFrameCheckerScale, columns=["# x", "y"]), ax = axs[1], x = "# x", y = "y").set(title="Checker")
# sns.scatterplot(pd.DataFrame(dFrameMoonsScale, columns=["# x", "y"]), ax = axs[2], x = "# x", y = "y").set(title="Moons") 

# plt.show()


## Задание 2 ------------------------------------------------------

# def ElbowMethod(dataFrame, frameTitle, axIdx):
#     inertias = {}

#     for countClusters in range(1,15):
#         kmeanModel = KMeans(n_clusters=countClusters)
#         kmeanModel.fit(dataFrame)
        
#         inertias[countClusters] = (kmeanModel.inertia_)

#     sns.lineplot(inertias, ax = axs[axIdx], marker='o').set(title=frameTitle)

# ElbowMethod(dFrameBlobsScale, "Blobs", 0) ### 5-7
# ElbowMethod(dFrameCheckerScale, "Checker", 1) ### 5-7
# ElbowMethod(dFrameMoonsScale, "Moons", 2) ### 11-14


# def SilhouetteMethod(dataFrame, frameTitle, axIdx):
#     silhouettes = {}

#     for countClusters in range(2,15):
#         kmeanModel = KMeans(n_clusters=countClusters)
        
#         silhouettes[countClusters] = (silhouette_score(dataFrame, kmeanModel.fit_predict(dataFrame)))

#     sns.lineplot(silhouettes, ax = axs[axIdx], marker='o').set(title=frameTitle)

# SilhouetteMethod(dFrameBlobsScale, "Blobs", 0) 
# SilhouetteMethod(dFrameCheckerScale, "Checker", 1) 
# SilhouetteMethod(dFrameMoonsScale, "Moons", 2) 

plt.show()



 


    