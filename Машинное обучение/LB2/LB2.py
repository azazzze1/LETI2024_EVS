import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.cluster.hierarchy import dendrogram, linkage

# Вариант 134Б 

# ЗАДАНИЕ 1 ---------------------------------------------------------

# dFrameBlobs = pd.read_csv("data\lab2_blobs.csv")
# dFrameChecker = pd.read_csv("data\lab2_checker.csv")
# dFrameMoons = pd.read_csv("data\lab2_noisymoons.csv")

# print(dFrameBlobs.describe())

# print(dFrameBlobs.head())
# print(dFrameChecker.head())
# print(dFrameMoons.head())

# fig, axs = plt.subplots(1, 3, sharex='col')

# sns.scatterplot(dFrameBlobs, ax = axs[0], x = "# x", y = "y").set(title="Blobs")
# sns.scatterplot(dFrameChecker, ax = axs[1], x = "# x", y = "y").set(title="Checker")
# sns.scatterplot(dFrameMoons, ax = axs[2], x = "# x", y = "y").set(title="Moons") 

# scaler = MinMaxScaler()
# scaler = StandardScaler()


# scaler.fit(dFrameBlobs) 
# dFrameBlobsScale = scaler.transform(dFrameBlobs)

# scaler.fit(dFrameChecker) 
# dFrameCheckerScale = scaler.transform(dFrameChecker)

# scaler.fit(dFrameMoons) 
# dFrameMoonsScale = scaler.transform(dFrameMoons)

# sns.scatterplot(pd.DataFrame(dFrameBlobsScale, columns=["# x", "y"]), ax = axs[0], x = "# x", y = "y").set(title="Blobs")
# sns.scatterplot(pd.DataFrame(dFrameCheckerScale, columns=["# x", "y"]), ax = axs[1], x = "# x", y = "y").set(title="Checker")
# sns.scatterplot(pd.DataFrame(dFrameMoonsScale, columns=["# x", "y"]), ax = axs[2], x = "# x", y = "y").set(title="Moons") 

# plt.show()


## ЗАДАНИЕ 2 ---------------------------------------------------------

## Задание 2.1 ----------------

# def ElbowMethod(dataFrame, frameTitle, axIdx):
#     inertias = {}

#     for countClusters in range(1,15):
#         kmeanModel = KMeans(n_clusters=countClusters)
#         kmeanModel.fit(dataFrame)
        
#         inertias[countClusters] = (kmeanModel.inertia_)

#     sns.lineplot(inertias, ax = axs[axIdx], marker='o').set(title=frameTitle)

# ElbowMethod(dFrameBlobsScale, "Blobs", 0) ### 5-7
# ElbowMethod(dFrameCheckerScale, "Checker", 1) ### 5-7
# ElbowMethod(dFrameMoonsScale, "Moons", 2) ### 10-14

## Задание 2.2 ----------------

# def SilhouetteMethod(dataFrame, frameTitle, axIdx):
#     silhouettes = {}

#     for countClusters in range(2,15):
#         kmeanModel = KMeans(n_clusters=countClusters)
        
#         silhouettes[countClusters] = (silhouette_score(dataFrame, kmeanModel.fit_predict(dataFrame)))

#     sns.lineplot(silhouettes, ax = axs[axIdx], marker='o').set(title=frameTitle)

# SilhouetteMethod(dFrameBlobsScale, "Blobs", 0) ## 5
# SilhouetteMethod(dFrameCheckerScale, "Checker", 1) ## 5
# SilhouetteMethod(dFrameMoonsScale, "Moons", 2) ## 10

## Задание 2.3 ----------------

# kmeansBlobs = KMeans(n_clusters = 5, 
# 			 init = 'random',
# 			 n_init = 10)

# clustIndexBlobs = kmeansBlobs.fit_predict(dFrameBlobsScale)
# centersBlobs = kmeansBlobs.cluster_centers_
# print("clust_index Blobs: ", clustIndexBlobs)
# print("censters Blobs: ", centersBlobs)
# print("Blobs inetrion: ", kmeansBlobs.inertia_) 


# kmeansChecker = KMeans(n_clusters = 5, 
# 			 init = 'random',
# 			 n_init = 10)

# clustIndexChecker = kmeansChecker.fit_predict(dFrameCheckerScale)
# centersChecker = kmeansChecker.cluster_centers_
# print("clust_index Checker: ", clustIndexChecker)
# print("censters Checker: ", centersChecker)
# print("Checker inetrion: ", kmeansChecker.inertia_) 


# kmeansMoons = KMeans(n_clusters = 10, 
# 			 init = 'random',
# 			 n_init = 10)

# clustIndexMoons = kmeansMoons.fit_predict(dFrameMoonsScale)
# centersMoons = kmeansMoons.cluster_centers_
# print("clust_index Moons: ", clustIndexMoons)
# print("censters Moons: ", centersMoons)
# print("Moons inetrion: ", kmeansMoons.inertia_) 

 
## Задание 2.4 ----------------

# dFrameBlobsKmeans = pd.DataFrame(dFrameBlobsScale, columns=["# x", "y"]).assign(cluster=clustIndexBlobs)
# dFrameCheckerKmeans = pd.DataFrame(dFrameCheckerScale, columns=["# x", "y"]).assign(cluster=clustIndexChecker)
# dFrameMoonsKmeans = pd.DataFrame(dFrameMoonsScale, columns=["# x", "y"]).assign(cluster=clustIndexMoons)

# sns.scatterplot(dFrameBlobsKmeans, ax = axs[0], x = "# x", y = "y", hue="cluster", palette="Set1").set(title="Blobs")
# sns.scatterplot(dFrameCheckerKmeans, ax = axs[1], x = "# x", y = "y", hue="cluster", palette="Set1").set(title="Checker")
# sns.scatterplot(dFrameMoonsKmeans, ax = axs[2], x = "# x", y = "y", hue="cluster", palette="Set1").set(title="Moons") 

# plt.show()


## Задание 2.5 ----------------

# vor = Voronoi(centersBlobs)
# figa = voronoi_plot_2d(vor, ax=axs[0], show_points = False)
# sns.scatterplot(dFrameBlobsKmeans, ax = axs[0], x = "# x", y = "y", hue="cluster", palette="Set1", legend=False).set(title="Blobs")
# sns.scatterplot(pd.DataFrame(centersBlobs, columns=["# x", "y"]), ax = axs[0], x = "# x", y = "y", marker = "x", color = "black", s = 150)


# vor = Voronoi(centersChecker)
# figa = voronoi_plot_2d(vor, ax=axs[1], show_points = False)
# sns.scatterplot(dFrameCheckerKmeans, ax = axs[1], x = "# x", y = "y", hue="cluster", palette="Set1", legend=False).set(title="Checker")
# sns.scatterplot(pd.DataFrame(centersChecker, columns=["# x", "y"]), ax = axs[1], x = "# x", y = "y", marker = "x", color = "black", s = 150)

# vor = Voronoi(centersMoons)
# figa = voronoi_plot_2d(vor, ax=axs[2], show_points = False)
# sns.scatterplot(dFrameMoonsKmeans, ax = axs[2], x = "# x", y = "y", hue="cluster", palette="Set1", legend=False).set(title="Moons") 
# sns.scatterplot(pd.DataFrame(centersMoons, columns=["# x", "y"]), ax = axs[2], x = "# x", y = "y", marker = "x", color = "black", s = 150)


## Задание 2.6 ----------------

# boxFig, boxAxs = plt.subplots(1, 2)

# sns.boxplot(dFrameBlobsKmeans, ax = boxAxs[0], x = "# x", hue="cluster", palette="Set1").set(title="Blobs X")
# sns.boxplot(dFrameBlobsKmeans, ax = boxAxs[1], x = "y", hue="cluster", palette="Set1").set(title="Blobs Y")

# sns.boxplot(dFrameCheckerKmeans, ax = boxAxs[0], x = "# x", hue="cluster", palette="Set1").set(title="Checker X")
# sns.boxplot(dFrameCheckerKmeans, ax = boxAxs[1], x = "y", hue="cluster", palette="Set1").set(title="Checker Y")

# sns.boxplot(dFrameMoonsKmeans, ax = boxAxs[0], x = "# x", hue="cluster", palette="Set1").set(title="Moons X")
# sns.boxplot(dFrameMoonsKmeans, ax = boxAxs[1], x = "y", hue="cluster", palette="Set1").set(title="Moons Y")

# plt.show()


## Задание 2.7 ----------------

# def describe27(dFrame): 
#     clusters = dFrame["cluster"].unique()
#     resultX = {}
#     resultY = {}
    
#     for clusterIdx in clusters: 
#         clustersPoints = dFrame[dFrame["cluster"] == clusterIdx]
#         resultX["count #" + str(clusterIdx)] = len(clustersPoints) 
#         resultY["count #" + str(clusterIdx)] = len(clustersPoints)
        
#         resultX["mean #" + str(clusterIdx)] = clustersPoints["# x"].mean()
#         resultY["mean #" + str(clusterIdx)] = clustersPoints["y"].mean()
        
#         resultX["std #" + str(clusterIdx)] = clustersPoints["# x"].std()
#         resultY["std #" + str(clusterIdx)] = clustersPoints["y"].std()
#         resultX["min #" + str(clusterIdx)] = clustersPoints["# x"].min()
#         resultY["min #" + str(clusterIdx)] = clustersPoints["y"].min()
        
#         resultX["max #" + str(clusterIdx)] = clustersPoints["# x"].max()
#         resultY["max #" + str(clusterIdx)] = clustersPoints["y"].max()
        
#     return resultX, resultY
    
# descrBlobsX, descrBlobsY = describe27(dFrameBlobsKmeans)
# descrCheckerX, descrCheckerY = describe27(dFrameCheckerKmeans)
# descrMoonsX, descrMoonsY = describe27(dFrameMoonsKmeans)

# print(descrCheckerY)



## ЗАДАНИЕ 3 ---------------------------------------------------------

# def kDistanceGraph(dFrame, dFrameTitle, axIdx):
#     neigh = NearestNeighbors(n_neighbors = 2)
#     nbrs = neigh.fit(dFrame)
#     distances, indices = nbrs.kneighbors(dFrame)
#     distances = (np.sort(distances, axis=0))[:, 1]
#     sns.lineplot(distances, ax=axs[axIdx]).set(title=dFrameTitle, ylabel = "Epsilon", xlabel = "Points sorted by distance to neighboor")
    

# kDistanceGraph(dFrameBlobsScale, "Blobs", 0) # 0.09 - 0.15
# kDistanceGraph(dFrameCheckerScale, "Checker", 1) # 0.10
# kDistanceGraph(dFrameMoonsScale, "Moons", 2) # 0.15 

# dbs = DBSCAN(eps=0.15, min_samples=5)
# clustersBlobs = dbs.fit_predict(dFrameBlobsScale)
# dFrameBlobsKmeans = pd.DataFrame(dFrameBlobsScale, columns=["# x", "y"]).assign(cluster=clustersBlobs)
# sns.scatterplot(dFrameBlobsKmeans, ax = axs[0], x = "# x", y = "y", hue="cluster", palette="Set1").set(title="Blobs")

# dbs = DBSCAN(eps=0.17, min_samples=6)
# clustersChecker = dbs.fit_predict(dFrameCheckerScale)
# dFrameCheckerKmeans = pd.DataFrame(dFrameCheckerScale, columns=["# x", "y"]).assign(cluster=clustersChecker)
# sns.scatterplot(dFrameCheckerKmeans, ax = axs[1], x = "# x", y = "y", hue="cluster", palette="Set1").set(title="Checker")

# dbs = DBSCAN(eps=0.20, min_samples=5)
# clustersMoons = dbs.fit_predict(dFrameMoonsScale)
# dFrameMoonsKmeans = pd.DataFrame(dFrameMoonsScale, columns=["# x", "y"]).assign(cluster=clustersMoons)
# sns.scatterplot(dFrameMoonsKmeans, ax = axs[2], x = "# x", y = "y", hue="cluster", palette="Set1").set(title="Moons")

# plt.show()


## ЗАДАНИЕ 4 ---------------------------------------------------------

# axs[0].set(title="Blobs")
# axs[1].set(title="Checker")
# axs[2].set(title="Moons")

# _linkage = "average"

# agc = AgglomerativeClustering(5, linkage = _linkage)
# ag_clust = agc.fit_predict(dFrameBlobsScale)
# dFrameBlobsAgc = pd.DataFrame(dFrameBlobsScale, columns=["# x", "y"]).assign(cluster=ag_clust)
# dendrogram(linkage(dFrameBlobsAgc, method=_linkage), ax = axs[0])
# sns.scatterplot(dFrameBlobsAgc, ax = axs[0], x = "# x", y = "y", hue="cluster", palette="Set1").set(title="Blobs")

# agc = AgglomerativeClustering(5, linkage = _linkage)
# ag_clust = agc.fit_predict(dFrameCheckerScale)
# dFrameCheckerAgc = pd.DataFrame(dFrameCheckerScale, columns=["# x", "y"]).assign(cluster=ag_clust)
# dendrogram(linkage(dFrameCheckerAgc, method=_linkage), ax = axs[1])
# sns.scatterplot(dFrameCheckerAgc, ax = axs[1], x = "# x", y = "y", hue="cluster", palette="Set1").set(title="Checker")

# agc = AgglomerativeClustering(7, linkage = _linkage)
# ag_clust = agc.fit_predict(dFrameMoonsScale)
# dFrameMoonsAgc = pd.DataFrame(dFrameMoonsScale, columns=["# x", "y"]).assign(cluster=ag_clust)
# dendrogram(linkage(dFrameMoonsAgc, method=_linkage))
# sns.scatterplot(dFrameMoonsAgc, ax = axs[2], x = "# x", y = "y", hue="cluster", palette="Set1").set(title="Moons")

# plt.show()


## ЗАДАНИЕ 5 ---------------------------------------------------------

dFrameWine = pd.read_csv("data\lab2_winequality_white.csv", sep=";")

# print(dFrameWine.head())

print(dFrameWine.describe())
print(dFrameWine.columns.to_list())

fig, axs = plt.subplots(4, 3, sharex='col')

axIdx = 0
for column in dFrameWine.columns.to_list(): 
    sns.histplot(dFrameWine, ax = axs[axIdx], bins=10, x = column, hue="quality", palette="Set1")
    axIdx+=1

## 

# print(dFrameWine.head()) 



