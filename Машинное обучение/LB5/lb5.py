from deap import base, algorithms
from deap import creator
from deap import tools
import numpy as np
import matplotlib.pyplot as plt
import random

import pandas as pd
import seaborn as sns

# константы генетического алгоритма
POPULATION_SIZE = 200 # количество индивидуумов в популяции
P_CROSSOVER = 0.9 # вероятность скрещивания
P_MUTATION = 0.1 # вероятность мутации индивидуума
MAX_GENERATIONS = 100 # максимальное количество поколений
HALL_OF_FAME_SIZE = 10

class City:
    def __init__(self, x : int, y : int, priorety : int, ):
        self.x = x
        self.y = y
        self.priorety = priorety

def loadDataTXT(filePath : str):
        cities = []
        file = open(filePath)
        countCities = int(file.readline())
        for i in range(countCities):
            x, y, priorety = file.readline().split(' ')
            cities.append(City(int(x),int(y),int(priorety))) 
        file.close()
        return(cities, countCities) 

cities, N = loadDataTXT("Машинное обучение\LB5\maps\map2.txt")

def generateIndividual(Individual, N):
    Individual = list(x for x in range(N-1))
    random.shuffle(Individual)
    return creator.Individual(Individual)
    
def oneMinFitness(individual):
    distance = lambda x0, y0, x1, y1 : np.sqrt((x0-x1)**2 + (y0-y1)**2)
    prioretyPenalty = lambda priorety0, priorety1 : 1000000 * np.heaviside(priorety1 - priorety0, 0)
    distancePenalty = lambda distance : (distance - 5) * np.heaviside(distance-5, 0)

    distStart = distance(cities[0].x, cities[0].y, cities[individual[0]+1].x, cities[individual[0]+1].y)
    distFinish = distance(cities[0].x, cities[0].y, cities[individual[N-2]+1].x, cities[individual[N-2]+1].y)
    metric = distStart + distFinish

    for i in range(N-2):
         dist = distance(cities[individual[i]+1].x, cities[individual[i]+1].y, cities[individual[i+1]+1].x, cities[individual[i+1]+1].y)
         metric += dist
         metric += prioretyPenalty(cities[individual[i]+1].priorety, cities[individual[i+1]+1].priorety)
         metric += distancePenalty(dist)

    metric += distancePenalty(distStart) + distancePenalty(distFinish)
    
    return metric,   

def draw(individual):
     X = [cities[0].x]
     Y = [cities[0].y]
     P = [cities[0].priorety]
     for i in individual:
          X.append(cities[i+1].x)
          Y.append(cities[i+1].y)
          P.append(cities[i+1].priorety)
     X.append(cities[0].x)
     Y.append(cities[0].y)
     P.append(cities[0].priorety)

     print(X)
     sns.lineplot(pd.DataFrame((X), columns=["X"]).assign(Y=Y), x="X", y="Y", sort=False, estimator=None, marker='o').set(title="Путь почтальона")
     plt.plot(cities[0].x, cities[0].y, color="red")
     plt.show()


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()

toolbox.register("individualCreator", generateIndividual, creator.Individual, N)
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

population = toolbox.populationCreator(n=POPULATION_SIZE)

toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=1/N)

toolbox.register("evaluate", oneMinFitness)

stats = tools.Statistics(lambda ind: ind.fitness.values)

stats.register("max", np.max)
stats.register("avg", np.mean)
stats.register("min", np.min)

hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

population, logbook = algorithms.eaSimple(population, toolbox, 
                                    cxpb=P_CROSSOVER,  
                                    mutpb=P_MUTATION, 
                                    ngen=MAX_GENERATIONS, 
                                    stats=stats,
                                    halloffame=hof,
                                    verbose=True)

maxFitnessValues, meanFitnessValues, minFitnessValues = logbook.select("max", "avg", "min")

print(hof.items[0], oneMinFitness(hof.items[0]))

draw(hof.items[0])

plt.plot(maxFitnessValues, color='red')
plt.plot(meanFitnessValues, color='green')
# plt.plot(minFitnessValues, color='blue')

plt.xlabel('Поколение')
plt.ylabel('Макс/средняя приспособленность')
plt.title('Зависимость максимальной и средней приспособленности от поколения')
plt.show()