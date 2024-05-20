import deap as dp

class City:
    def __init__(self, x : int, y : int, priorety : int, ):
        self.x = x
        self.y = y
        self.priorety = priorety

    def printInfo(self):
        return ("x = " + str(self.x) + "\ny = " + str(self.y) + "\npriorety = " + str(self.priorety))
       
class Map:
    def __init__(self):
        self.cities = []

    def loadMapTXT(self, filePath : str):
        self.cities.clear()

        file = open(filePath)
        countCities = int(file.readline())
        for i in range(countCities):
            x, y, priorety = file.readline().split(' ')
            self.cities.append(City(x,y,priorety))

    def printCities(self):
        for i in range(len(self.cities)):
            print("City #" + str(i) + ":\n" + self.cities[i].printInfo())
        
map = Map()
map.loadMapTXT("E:\LETI2024_EVS\Машинное обучение\LB5\maps\map1.txt")
  



                 
    

