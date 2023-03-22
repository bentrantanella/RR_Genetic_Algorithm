import math

import numpy as np
import random
import matplotlib.pyplot as plt
import csv




class HillClimber:
    def __init__(self, mutationRate):
        self.mutationRate = mutationRate

    def mutate(self, child):
        childList = list(child)

        for i in range(64):
            if random.random() < self.mutationRate:
                if childList[i] == '1':
                    childList[i] = '*'
                else:
                    childList[i] = '1'

        str1 = ""
        newChild = str1.join(childList)
        return newChild

    def calcFitness(self, s):
        score = 0

        #calc 8s
        if s[0:8].count('1') == 8:
            score+=8
        if s[8:16].count('1') == 8:
            score+=8
        if s[16:24].count('1') == 8:
            score+=8
        if s[24:32].count('1') == 8:
            score+=8
        if s[32:40].count('1') == 8:
            score+=8
        if s[40:48].count('1') == 8:
            score+=8
        if s[48:56].count('1') == 8:
            score+=8
        if s[56:64].count('1') == 8:
            score+=8

        #calc 16s
        if s[0:16].count('1') == 16:
            score+=16
        if s[16:32].count('1') == 16:
            score+=16
        if s[32:48].count('1') == 16:
            score+=16
        if s[48:64].count('1') == 16:
            score+=16

        #calc 32s
        if s[0:32].count('1') == 32:
            score+=32
        if s[32:64].count('1') == 32:
            score+=32

        #calc 64
        if s.count('1') == 64:
            score+=64

        #0 fitness case
        if score == 0:
            score += 1

        return score

    def runNTrials(self, n):
        numGenTracker = []

        for i in range(n):
            numGen = self.runTrial()
            numGenTracker.append(numGen)
            print("Trial #" + str(i + 1) + ": " + numGen + " generations")

        avgGen = sum(numGenTracker) // len(numGenTracker)
        return avgGen

    def runTrial(self):
        currentBitString = ""
        for i in range(64):
            if random.random() < 0.5:
                currentBitString += "1"
            else:
                currentBitString += "*"

        numGen = 0
        while currentBitString != "1111111111111111111111111111111111111111111111111111111111111111":
            newBitString = self.mutate(currentBitString)
            #print(newBitString + ", " + str(self.calcFitness(newBitString)))
            if self.calcFitness(newBitString) > self.calcFitness(currentBitString):
                currentBitString = newBitString
                print(currentBitString + ", " + str(self.calcFitness(currentBitString)))
                print(numGen)
            numGen += 1


        return numGen



class Chromosome:

    def __init__(self, fitness, bitstring):
        self.bitstring = bitstring
        self.fitness = fitness

    def __str__(self):
        return str(self.fitness) + ", " + self.bitstring


class RoyalRoad:
    def __init__(self, populationSize, crossoverRate, mutationRate):
        self.populationSize = populationSize
        self.crossoverRate = crossoverRate
        self.mutationRate = mutationRate

    def createFirstGen(self):
        firstGen = []

        for i in range(128):
            c = ""
            for j in range(64):
                if random.random() < 0.5:
                    c += "1"
                else:
                    c += "*"
            fit = self.calcFitness(c)
            child = Chromosome(fit, c)
            firstGen.append(child)

        return firstGen

    def calcFitness(self, s):
        score = 0

        #calc 8s
        if s[0:8].count('1') == 8:
            score+=8
        if s[8:16].count('1') == 8:
            score+=8
        if s[16:24].count('1') == 8:
            score+=8
        if s[24:32].count('1') == 8:
            score+=8
        if s[32:40].count('1') == 8:
            score+=8
        if s[40:48].count('1') == 8:
            score+=8
        if s[48:56].count('1') == 8:
            score+=8
        if s[56:64].count('1') == 8:
            score+=8

        #calc 16s
        if s[0:16].count('1') == 16:
            score+=16
        if s[16:32].count('1') == 16:
            score+=16
        if s[32:48].count('1') == 16:
            score+=16
        if s[48:64].count('1') == 16:
            score+=16

        #calc 32s
        if s[0:32].count('1') == 32:
            score+=32
        if s[32:64].count('1') == 32:
            score+=32

        #calc 64
        if s.count('1') == 64:
            score+=64

        #0 fitness case
        if score == 0:
            score += 1

        return score


    def createChildren(self, p1, p2, crossType):
        c1 = ""
        c2 = ""
        if random.random() < self.crossoverRate:
            if crossType == 0:
                c1 += self.singlePCrossover(p1.bitstring, p2.bitstring)
                c2 += self.singlePCrossover(p2.bitstring, p1.bitstring)
            elif crossType == 1:
                c1 += self.doublePCrossover(p1.bitstring, p2.bitstring)
                c2 += self.doublePCrossover(p2.bitstring, p1.bitstring)
            else:
                c1 += self.uniformCrossover(p1.bitstring, p2.bitstring)
                c2 += self.uniformCrossover(p1.bitstring, p2.bitstring)

        else:
            c1 += p1.bitstring
            c2 += p2.bitstring

        c1 = self.mutate(c1)
        c2 = self.mutate(c2)

        child1 = Chromosome(self.calcFitness(c1), c1)
        child2 = Chromosome(self.calcFitness(c2), c2)
        newChildren = (child1, child2)

        return newChildren

    def singlePCrossover(self, parent1, parent2):
        crossPoint = random.randint(0, 63)
        child = parent1[:crossPoint] + parent2[crossPoint:]
        return child

    def doublePCrossover(self, parent1, parent2):
        cPoint1 = random.randint(0, 63)
        cPoint2 = random.randint(0, 63)

        if cPoint2 < cPoint1:
            temp = cPoint1
            cPoint1 = cPoint2
            cPoint2 = temp

        child = parent1[0:cPoint1] + parent2[cPoint1:cPoint2] + parent1[cPoint2:64]
        return child

    def uniformCrossover(self, parent1, parent2):
        child = ""
        p1List = list(parent1)
        p2List = list(parent2)
        for i in range(64):
            if random.random() < 0.5:
                child += p1List[i]
            else:
                child += p2List[i]

        return child


    def mutate(self, child):
        childList = list(child)

        for i in range(64):
            if random.random() < self.mutationRate:
                if childList[i] == '1':
                    childList[i] = '*'
                else:
                    childList[i] = '1'

        str1 = ""
        newChild = str1.join(childList)
        return newChild

    def pickParent(self, currentGen):
        totalFit = 0

        for p in currentGen:
            totalFit += p.fitness

        pick = random.randint(1, totalFit)

        runningTot = 0
        for p in currentGen:
            runningTot += p.fitness
            if runningTot >= pick:
                return p


    def checkIfDone(self, currentGen):
        perfectString = "1111111111111111111111111111111111111111111111111111111111111111"
        isPerfect = False

        for c in currentGen:
            if c.bitstring == perfectString:
                isPerfect = True
                break

        return isPerfect

    def runTrial(self, crossType):
        currentGen = self.createFirstGen()
        genCounter = 0

        while not self.checkIfDone(currentGen):
            nextGen = []

            while len(nextGen) < 128:
                p1 = self.pickParent(currentGen)
                p2 = self.pickParent(currentGen)

                newChildren = self.createChildren(p1, p2, crossType)
                nextGen.append(newChildren[0])
                nextGen.append(newChildren[1])

            currentGen = nextGen
            genCounter += 1

            highFit = 0
            highP = currentGen[0]
            for p in currentGen:
                if p.fitness > highFit:
                    highFit = p.fitness
                    highP = p

            #print(highP)

        return genCounter

    def runNTimes(self, n, crossType):
        numGenTracker = []
        minGen = math.inf
        maxGen = math.inf * -1

        for i in range(n):
            numGen = self.runTrial(crossType)
            if numGen > maxGen:
                maxGen = numGen
            if numGen < minGen:
                minGen = numGen
            numGenTracker.append(numGen)
            print("Trial #" + str(i + 1) + ": " + str(numGen) + " generations")

        avgGen = sum(numGenTracker) // len(numGenTracker)
        info = [maxGen, minGen, avgGen]
        return info

class Selection:

    def __init__(self, selecList):
        self.selecList = selecList

    def SortPop(self, p):
        '''
        presuming that a population is a list of tuples of the form
        (fitness,genome)
        sorts the population from most to least fit
        '''
        # using different functions to sort lists is kind of cool
        p.sort(key=lambda x: x.fitness, reverse=True)
        return p

    def Make_Roulette(self, pop):
        '''
        Uses the prefix_scan (cumulative sum) algorithm
        to come up with a set of roulette wheel slots
        any individual's chance of being selected is
        their fitness/ sum of all fitnesses
        '''
        # sort  pop
        pop = self.SortPop(pop)
        # gets fitnesses out
        fits = np.array([p.fitness for p in pop])
        # gets genes out
        genes = [p.bitstring for p in pop]
        # divides every fitness by sum of all fitnesses
        fits = fits / sum(fits)
        # calculates prefix sum
        prefix_sum = np.cumsum(fits)
        # print(prefix_sum)
        # zips prefixes and genes back together to return
        return list(zip(prefix_sum, genes))

    def Pick_From_Roulette(self, wheel):
        '''
        Given a roulette wheel generated by Make_Roulette
        returns an individual from that wheel.

        '''
        prob = np.random.uniform()
        for p, g in wheel:
            # print("prob:", prob," p: ", p)
            if prob <= p:
                return g
        #print("prob:", prob, " last: ", list(wheel))
        #print("i should not have gotten here")
        return None

    '''
    passing in a list of tuples in the form (fitness,genome) and randomly picking one
    '''
    def new_selection(self):
        wheel = self.Make_Roulette(self.selecList)
        return self.Pick_From_Roulette(wheel)


def main():

    singleMaxList = []
    singleMinList = []
    singleAvgList = []

    doubleMaxList = []
    doubleMinList = []
    doubleAvgList = []

    uniformMaxList = []
    uniformMinList = []
    uniformAvgList = []

    fields = ["Run Number", "Single Point Max", "Double Point Max", "Uniform Max", "Single Point Min",
              "Double Point Min", "Uniform Min", "Single Point Avg", "Double Point Avg", "Uniform Avg"]
    allInfo = []
    allInfo.append(fields)

    for i in range(30):
        r = RoyalRoad(128, 0.7, 0.005)
        runInfo = []

        SPGenInfo = r.runNTimes(100, 0)
        singleMaxList.append(SPGenInfo[0])
        singleMinList.append(SPGenInfo[1])
        singleAvgList.append(SPGenInfo[2])
        print("Average number of generations for single point crossover RR: " + str(SPGenInfo[2]))

        DPGenInfo = r.runNTimes(100, 1)
        doubleMaxList.append(DPGenInfo[0])
        doubleMinList.append(DPGenInfo[1])
        doubleAvgList.append(DPGenInfo[2])
        print("Average number of generations for double point crossover RR: " + str(DPGenInfo[2]))

        UGenInfo = r.runNTimes(100, 2)
        uniformMaxList.append(UGenInfo[0])
        uniformMinList.append(UGenInfo[1])
        uniformAvgList.append(UGenInfo[2])
        print("Average number of generations for uniform crossover RR: " + str(UGenInfo[2]))
        print("Run number: " + str(i + 1))

        runInfo.append(i + 1)
        runInfo.append(SPGenInfo[0])
        runInfo.append(DPGenInfo[0])
        runInfo.append(UGenInfo[0])
        runInfo.append(SPGenInfo[1])
        runInfo.append(DPGenInfo[1])
        runInfo.append(UGenInfo[1])
        runInfo.append(SPGenInfo[2])
        runInfo.append(DPGenInfo[2])
        runInfo.append(UGenInfo[2])

        allInfo.append(runInfo)

    plt.plot(range(1, 31), singleAvgList, color='tab:blue')
    plt.plot(range(1, 31), doubleAvgList, color='tab:green')
    plt.plot(range(1, 31), uniformAvgList, color='tab:orange')
    plt.title("Average number of generations")
    plt.xlabel("Run number")
    plt.ylabel("Num of Gens")
    legend_drawn_flag = True
    plt.legend(["single", "double", "uniform"], loc=0, frameon=legend_drawn_flag)
    plt.show()

    plt.plot(range(1, 31), singleMaxList, color='tab:blue')
    plt.plot(range(1, 31), doubleMaxList, color='tab:green')
    plt.plot(range(1, 31), uniformMaxList, color='tab:orange')
    plt.title("Max number of generations")
    plt.xlabel("Run number")
    plt.ylabel("Num of Gens")
    legend_drawn_flag = True
    plt.legend(["single", "double", "uniform"], loc=0, frameon=legend_drawn_flag)
    plt.show()

    plt.plot(range(1, 31), singleMinList, color='tab:blue')
    plt.plot(range(1, 31), doubleMinList, color='tab:green')
    plt.plot(range(1, 31), uniformMinList, color='tab:orange')
    plt.title("Min number of generations")
    plt.xlabel("Run number")
    plt.ylabel("Num of Gens")
    legend_drawn_flag = True
    plt.legend(["single", "double", "uniform"], loc=0, frameon=legend_drawn_flag)
    plt.show()

    filename = "GA_AllRunInfo.csv"
    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(allInfo)




    #h = HillClimber(0.5)
    #avgHCGens = h.runNTrials(1)
    #print("Average number of generations for HC: " + str(avgHCGens))

if __name__ == '__main__':
    main()
