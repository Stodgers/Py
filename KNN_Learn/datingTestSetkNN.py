from numpy import *
import matplotlib
import matplotlib.pyplot as plt
def file2matrix(filename):
    fr = open(filename)
    arryline = fr.readlines()
    numberofline = len(arryline)
    returnmat = zeros((numberofline,3))
    classLabelvector = []
    index = 0
    for line in arryline:
        line = line.strip()
        listFrom = line.split('\t')
        returnmat[index,:] = listFrom[0:3]
        classLabelvector.append(int(listFrom[-1]))
        index += 1
    return returnmat,classLabelvector
def classify0(inX, dataSet, labels, k):
    # 获取样本数据数量
    dataSetSize = dataSet.shape[0]   #训练集总和  行数！！

    # 矩阵运算，计算测试数据与每个样本数据对应数据项的差值
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet #把测试样例和每一行对比

    # sqDistances 上一步骤结果平方和
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)

    # 取平方根，得到距离向量
    distances = sqDistances ** 0.5

    # 按照距离从低到高排序
    sortedDistIndicies = distances.argsort()
    classCount = {}

    # 依次取出最近的样本数据
    for i in range(k):
        # 记录该样本数据所属的类别
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    # 对类别出现的频次进行排序，从高到低
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    # 返回出现频次最高的类别
    return sortedClassCount[0][0]

datamat,datalable = file2matrix('datingTEstSetkNN\datingTestSet2.txt')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datamat[:,1],datamat[:,2],15.0*array(datalable),15.0*array(datalable))
plt.show()
from pylab import *
from mpl_toolkits.mplot3d import Axes3D

fig = figure()
ax = Axes3D(fig)
X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot')

show()