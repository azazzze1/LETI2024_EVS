import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, SGDRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error

dFrameLin = pd.read_csv("LB3\src\lab3_lin2.csv")

# print(dFrameLin.head())

dFrameLinTrain, dFrameLinTest = train_test_split(dFrameLin, random_state=42)

# fig, axs = plt.subplots(2, 3, sharex='col')

# for axIdx in range (3):    
#     sns.scatterplot(dFrameLinTest, ax = axs[0][axIdx], x = "x" + str(axIdx + 1), y = "y").set(title="Test")
#     sns.scatterplot(dFrameLinTrain, ax = axs[1][axIdx], x = "x" + str(axIdx + 1), y = "y").set(title="Train")   

# plt.show()


# ЗАДАНИЕ 1.3 ------------------------------------------------- Модель ориентировалась на x3, т.к. он линейный 

trainX = dFrameLinTrain.iloc[:,0:3]
trainY = dFrameLinTrain.iloc[:,3:4]

testX = dFrameLinTest.iloc[:,0:3]
testY = dFrameLinTest.iloc[:,3:4]

linReg = LinearRegression()

linReg.fit(trainX, trainY)
predictTestY = linReg.predict(testX)

dFrameLinPredict = testX.reset_index(drop= True).assign(y=pd.DataFrame(predictTestY))

# print("Коэффициенты регрессии: ", linReg.coef_)

# X = [x/100 for x in range(-200,100,1)]

# y1 = [linReg.coef_[0][2] * x + linReg.intercept_ for x in X]

# print(type(X[0]), type(y1[0]))
# plt.scatter(X, y1)
# sns.scatterplot(dFrameLinTest, x = "x3", y = "y")
# plt.show()

fig, axs = plt.subplots(1, 3, sharex='col')

# for axIdx in range (3):    
#     sns.scatterplot(dFrameLinTest, ax = axs[axIdx-2], x = "x" + str(axIdx + 1), y = "y")
#     sns.scatterplot(dFrameLinPredict, ax = axs[axIdx-2], x = "x" + str(axIdx + 1), y = "y") 

# plt.show()



# ЗАДАНИЕ 1.4 -------------------------------------------------

# linReg.fit(trainX, trainY)
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


# ЗАДАНИЕ 1.5 -------------------------------------------------


def linModelTask(linModel, nameModel):
    linModel.fit(trainX, trainY) 
    predictTestY = linModel.predict(testX)
    predictTrainY = linModel.predict(trainX)

    dFrameLinPredict = testX.reset_index(drop= True).assign(y=pd.DataFrame(predictTestY))

    print(nameModel+ " - Коэффициенты регрессии: ", linReg.coef_)

    TrainR2 = r2_score(trainY, predictTrainY)
    TestR2 = r2_score(testY, predictTestY)

    print(nameModel + " - R2 (train - test): ", TrainR2, TestR2)

    TrainMAE = mean_absolute_error(trainY, predictTrainY)
    TestMAE = mean_absolute_error(testY, predictTestY)

    print(nameModel + " - MAE (train - test): ", TrainMAE, TestMAE)

    TrainMAPE = mean_absolute_percentage_error(trainY, predictTrainY)
    TestMAPE = mean_absolute_percentage_error(testY, predictTestY)

    print(nameModel + " - MAPE (train - test): ", TrainMAPE, TestMAPE)

    for axIdx in range (3):    
        sns.scatterplot(dFrameLinTest, ax = axs[axIdx], x = "x" + str(axIdx + 1), y = "y").set(title=nameModel)
        sns.scatterplot(dFrameLinPredict, ax = axs[axIdx], x = "x" + str(axIdx + 1), y = "y").set(title=nameModel) 

    plt.show()



#Lasso
linReg = Lasso(alpha=0.1)
linModelTask(linReg, "Lasso")

# linReg = Ridge(alpha=0.1)
# linModelTask(linReg, "Ridge")

# linReg = ElasticNet(alpha=0.1)
# linModelTask(linReg, "ElasticNet")

# linReg = SGDRegressor(alpha=0.1)
# linModelTask(linReg, "ElasticNet")