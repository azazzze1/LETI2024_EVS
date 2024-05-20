import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, SGDRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder

# dFrameLin = pd.read_csv("E:\LETI2024_EVS\Машинное обучение\LB3\src\lab3_lin2.csv")

# print(dFrameLin.head())

# dFrameLinTrain, dFrameLinTest = train_test_split(dFrameLin, random_state=42)

# fig, axs = plt.subplots(2, 3, sharex='col')

# for axIdx in range (3):    
#     sns.scatterplot(dFrameLinTest, ax = axs[0][axIdx], x = "x" + str(axIdx + 1), y = "y").set(title="Test")
#     sns.scatterplot(dFrameLinTrain, ax = axs[1][axIdx], x = "x" + str(axIdx + 1), y = "y").set(title="Train")   

# plt.show()


# ЗАДАНИЕ 1.3 ------------------------------------------------- Модель ориентировалась на x3, т.к. он линейный 

# trainX = dFrameLinTrain.iloc[:,0:3]
# trainY = dFrameLinTrain.iloc[:,3:4]

# testX = dFrameLinTest.iloc[:,0:3]
# testY = dFrameLinTest.iloc[:,3:4]

# linReg = LinearRegression()

# linReg.fit(trainX, trainY)
# predictTestY = linReg.predict(testX)

# dFrameLinPredict = testX.reset_index(drop= True).assign(y=pd.DataFrame(predictTestY))

# print("Коэффициенты регрессии: ", linReg.coef_)

# X = [x/100 for x in range(-200,100,1)]

# y1 = [linReg.coef_[0][2] * x + linReg.intercept_ for x in X]

# print(type(X[0]), type(y1[0]))
# plt.scatter(X, y1)
# sns.scatterplot(dFrameLinTest, x = "x3", y = "y")
# plt.show()

# fig, axs = plt.subplots(1, 3, sharex='col')

# for axIdx in range (3):    
#     sns.scatterplot(dFrameLinTest, ax = axs[axIdx-2], x = "x" + str(axIdx + 1), y = "y", color="red")
#     sns.scatterplot(dFrameLinPredict, ax = axs[axIdx-2], x = "x" + str(axIdx + 1), y = "y", color="blue") 

# plt.show()



# ЗАДАНИЕ 1.4 -------------------------------------------------

# predictTrainY = linReg.predict(trainX)

# TrainR2 = r2_score(trainY, predictTrainY)
# TestR2 = r2_score(testY, predictTestY)

# print("R2 (train - test): ", TrainR2, TestR2)

# TrainMAE = mean_absolute_error(trainY, predictTrainY)
# TestMAE = mean_absolute_error(testY, predictTestY)

# print("MAE (train - test): ", TrainMAE, TestMAE)

# TrainMAPE = mean_absolute_percentage_error(trainY, predictTrainY)
# TestMAPE = mean_absolute_percentage_error(testY, predictTestY)

# print("MAPE (train - test): ", TrainMAPE, TestMAPE)



# ЗАДАНИЕ 2.1 -------------------------------------------------

# dFramePoly = pd.read_csv("E:\LETI2024_EVS\Машинное обучение\LB3\src\lab3_poly2.csv")

# print(dFramePoly.head())



# ЗАДАНИЕ 2.2 -------------------------------------------------

# dFramePolyTrain, dFramePolyTest = train_test_split(dFramePoly, random_state=42)

# fig, axs = plt.subplots(1, 2, sharex='col')

# sns.scatterplot(dFramePolyTest, ax = axs[0], x = "x", y = "y").set(title="Test")
# sns.scatterplot(dFramePolyTrain, ax = axs[1], x = "x", y = "y").set(title="Train")   

# plt.show()



# ЗАДАНИЕ 2.3 -------------------------------------------------

# trainX = dFramePolyTrain.iloc[:,0:1]
# trainY = dFramePolyTrain.iloc[:,1:2]

# testX = dFramePolyTest.iloc[:,0:1]
# testY = dFramePolyTest.iloc[:,1:2]

# linReg = LinearRegression()

# linReg.fit(trainX, trainY)
# predictTestY = linReg.predict(testX)

# dFramePolyPredict = testX.reset_index(drop= True).assign(y=pd.DataFrame(predictTestY))

# print("Коэффициенты регрессии: ", linReg.coef_)

# sns.scatterplot(dFramePolyTest, x = "x", y = "y")
# sns.scatterplot(dFramePolyPredict, x = "x", y = "y") 

# plt.show()



# ЗАДАНИЕ 2.4 -------------------------------------------------

# def PolyFeatures(k):
#     poly = PolynomialFeatures(degree=k)
#     polyTrainX = poly.fit_transform(trainX)
#     polyTestX = poly.transform(testX)

#     linReg = LinearRegression()
#     linReg.fit(polyTrainX, trainY)

#     predictTestY = linReg.predict(polyTestX)
#     predictTrainY = linReg.predict(polyTrainX)

#     TrainR2 = r2_score(trainY, predictTrainY)
#     TestR2 = r2_score(testY, predictTestY)

#     return [TrainR2, TestR2]


# K = [k for k in range(1, 15)]

# sns.lineplot(pd.DataFrame([PolyFeatures(k)[0] for k in K], columns=["r2"]).assign(degree=K), x = "degree", y = "r2", color="red")
# sns.lineplot(pd.DataFrame([PolyFeatures(k)[1] for k in K], columns=["r2"]).assign(degree=K), x = "degree", y = "r2", color="blue")

# plt.show()



# ЗАДАНИЕ 2.5 -------------------------------------------------


# poly = PolynomialFeatures(degree=6)
# polyTrainX = poly.fit_transform(trainX)
# polyTestX = poly.transform(testX)

# linReg = LinearRegression()
# linReg.fit(polyTrainX, trainY)

# predictTestY = linReg.predict(polyTestX)
# predictTrainY = linReg.predict(polyTrainX)

# TrainR2 = r2_score(trainY, predictTrainY)
# TestR2 = r2_score(testY, predictTestY)

# print("R2 (train - test): ", TrainR2, TestR2)

# TrainMAE = mean_absolute_error(trainY, predictTrainY)
# TestMAE = mean_absolute_error(testY, predictTestY)

# print("MAE (train - test): ", TrainMAE, TestMAE)

# TrainMAPE = mean_absolute_percentage_error(trainY, predictTrainY)
# TestMAPE = mean_absolute_percentage_error(testY, predictTestY)

# print("MAPE (train - test): ", TrainMAPE, TestMAPE)


# ЗАДАНИЕ 2.6 -------------------------------------------------


# dFramePolyPredict = testX.reset_index(drop= True).assign(y=pd.DataFrame(predictTestY))
# sns.lineplot(dFramePolyPredict, x = "x", y = "y", color="red")
# sns.scatterplot(dFramePolyTest, x = "x", y = "y")

# plt.show()



# ЗАДАНИЕ 3.1 -------------------------------------------------

dFrameStud = pd.read_csv("E:\LETI2024_EVS\Машинное обучение\LB3\src\Student_Performance.csv")

# print(dFrameStud.head())



# ЗАДАНИЕ 3.2 -------------------------------------------------

le = LabelEncoder()
le.fit(dFrameStud["Extracurricular Activities"])

dFrameStud["Extracurricular Activities"] = le.transform(dFrameStud["Extracurricular Activities"])

# print(dFrameStud["Extracurricular Activities"].head())

# print("Количество записей до удаления дубликатов: ", len(dFrameStud))

dFrameStud = dFrameStud.drop_duplicates()

# print("Количество записей после удаления дубликатов: ", len(dFrameStud))

# print(dFrameStud.info())

dFrameStudTrain, dFrameStudTest = train_test_split(dFrameStud, random_state=42)



# ЗАДАНИЕ 3.3 -------------------------------------------------

trainX = dFrameStudTrain.iloc[:,0:5]
trainY = dFrameStudTrain.iloc[:,5:6]

testX = dFrameStudTest.iloc[:,0:5]
testY = dFrameStudTest.iloc[:,5:6]

linReg = LinearRegression()
linReg.fit(trainX, trainY)

predictTestY = linReg.predict(testX) 
predictTrainY = linReg.predict(trainX) 

dFrameStudPredict = testX.reset_index(drop= True).assign(Performance_Index=pd.DataFrame(predictTestY))

print("Коэффициенты регрессии: ", linReg.coef_)

# ЗАДАНИЕ 3.4 -------------------------------------------------

TrainR2 = r2_score(trainY, predictTrainY)
TestR2 = r2_score(testY, predictTestY)

print("R2 (train - test): ", TrainR2, TestR2)

TrainMAE = mean_absolute_error(trainY, predictTrainY)
TestMAE = mean_absolute_error(testY, predictTestY)

print("MAE (train - test): ", TrainMAE, TestMAE)

TrainMAPE = mean_absolute_percentage_error(trainY, predictTrainY)
TestMAPE = mean_absolute_percentage_error(testY, predictTestY)

print("MAPE (train - test): ", TrainMAPE, TestMAPE)




fig, axs = plt.subplots(4, 2, sharex='col')

sns.scatterplot(dFrameStudPredict, ax=axs[0][0], x = "Hours Studied", y = "Performance_Index", color="red")
sns.scatterplot(dFrameStudPredict, ax=axs[0][1], x = "Previous Scores", y = "Performance_Index", color="red")
sns.scatterplot(dFrameStudPredict, ax=axs[1][0], x = "Extracurricular Activities", y = "Performance_Index", color="red")
sns.scatterplot(dFrameStudPredict, ax=axs[2][0], x = "Sleep Hours", y = "Performance_Index", color="red")
sns.scatterplot(dFrameStudPredict, ax=axs[3][0], x = "Sample Question Papers Practiced", y = "Performance_Index", color="red")

sns.scatterplot(dFrameStudTest, ax=axs[0][0], x = "Hours Studied", y = "Performance Index", color="blue")
sns.scatterplot(dFrameStudTest, ax=axs[0][1], x = "Previous Scores", y = "Performance Index", color="blue")
sns.scatterplot(dFrameStudTest, ax=axs[1][0], x = "Extracurricular Activities", y = "Performance Index", color="blue")
sns.scatterplot(dFrameStudTest, ax=axs[2][0], x = "Sleep Hours", y = "Performance Index", color="blue")
sns.scatterplot(dFrameStudTest, ax=axs[3][0], x = "Sample Question Papers Practiced", y = "Performance Index", color="blue")

plt.show()