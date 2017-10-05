from math import log
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelComtents = {}
    for featVec in dataSet:
        currentLanbel = featVec[-1]
        if currentLanbel not in labelComtents.keys():
            labelComtents[currentLanbel] = 0
        labelComtents[currentLanbel] += 1
    shannonEnt = 0.0
    for key in labelComtents:
        prob =float(labelComtents[key])/numEntries
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels
mydata ,labels= createDataSet()
mydata[0][-1]='maybe'
#print(calcShannonEnt(mydata))
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]     #chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet
