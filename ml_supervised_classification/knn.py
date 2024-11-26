"""
首先，我们将探讨k-近邻算法的基本理论，以及如何使用距离测量的方法分类物品；
其次我们将使用Python从文本文件中导入并解析数据；
再次，本书讨论了当存在许多数据来源时，如何避免计算距离时可能碰到的一些常见错误；
最后，利用实际的例子讲解如何使用k-近邻算法改进约会网站和手写数字识别系统。


k-近邻算法概述
    k-近邻算法采用测量不同特征值之间的距离方法进行分类。

    优点：精度高、对异常值不敏感、无数据输入假定。
    缺点：计算复杂度高、空间复杂度高。
    适用数据范围：数值型和标称型。

    工作原理是：
        存在一个样本数据集合，也称作训练样本集，并且样本集中每个数据都存在标签，即我们知道样本集中每一数据与所属分类的对应关系。
        输入没有标签的新数据后，将新数据的每个特征与样本集中数据对应的特征进行比较，然后算法提取样本集中特征最相似数据（最近邻）的分类标签。
            一般来说，我们只选择样本数据集中前k个最相似的数据，这就是k-近邻算法中k的出处，通常k是不大于20的整数。 
        最后，选择k个最相似数据中出现次数最多的分类，作为新数据的分类。

    k-近邻算法的一般流程
        (1) 收集数据：可以使用任何方法。 
        (2) 准备数据：距离计算所需要的数值，最好是结构化的数据格式。 
        (3) 分析数据：可以使用任何方法。 
        (4) 训练算法：此步骤不适用于k-近邻算法。 
        (5) 测试算法：计算错误率。
        (6) 使用算法：首先需要输入样本数据和结构化的输出结果，然后运行k-近邻算法判定输入数据分别属于哪个分类，最后应用对计算出的分类执行后续的处理。
"""

#准备：使用 Python 导入数据
from numpy import *
import operator

def createDataset():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]]) 
    labels = ['A', 'A', 'B', 'B']
    return group, labels

"""
实施 kNN 算法
    对未知类别属性的数据集中的每个点依次执行以下操作：
    (1) 计算已知类别数据集中的点与当前点之间的距离；
    (2) 按照距离递增次序排序；
    (3) 选取与当前点距离最小的k个点；
    (4) 确定前k个点所在类别的出现频率；
    (5) 返回前k个点出现频率最高的类别作为当前点的预测分类。
"""
def classify0(inX, dataSet, labels, k):
    #距离计算,使用欧氏距离公式
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()     
    #选择距离最小的k个点
    classCount={}          
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    #排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

"""
为了预测数据所在分类，在Python提示符中输入下列命令：
classify0([0, 0], group, labels, 3)
"""

"""
如何测试分类器
    此外分类器的性能也会受到多种因素的影响，如分类器设置和数据集等。
    不同的算法在不同数据集上的表现可能完全不同，这也是本部分的6章都在讨论分类算法的原因所在。
    通过大量的测试数据，我们可以得到分类器的错误率——分类器给出错误结果的次数除以测试执行的总数。
        错误率是常用的评估方法，主要用于评估分类器在某个数据集上的执行效果。

上一节介绍的例子已经可以正常运转了，但是并没有太大的实际用处，本章的后两节将在现实世界中使用k-近邻算法。
"""

"""
示例：使用 k-近邻算法改进约会网站的配对效果
    (1) 收集数据：提供文本文件。
    (2) 准备数据：使用Python解析文本文件。 
    (3) 分析数据：使用Matplotlib画二维扩散图。 
    (4) 训练算法：此步骤不适用于k-近邻算法。
    (5) 测试算法：使用海伦提供的部分数据作为测试样本。
            测试样本和非测试样本的区别在于：测试样本是已经完成分类的数据，如果预测分类与实际类别不同，则标记为一个错误。 
    (6) 使用算法：产生简单的命令行程序，然后海伦可以输入一些特征数据以判断对方是否为自己喜欢的类型。
"""

"""
准备数据：从文本文件中解析数据
    把这些数据存放在文本文件datingTestSet.txt中，每个样本数据占据一行，总共有1000行。
    在将上述特征数据输入到分类器之前，必须将待处理数据的格式改变为分类器可以接受的格式。
    在kNN.py中创建名为file2matrix的函数，以此来处理输入格式问题。
        该函数的输入为文件名字符串，输出为训练样本矩阵和类标签向量。
"""

def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())         #get the number of lines in the file
    """
    为了简化处理，我们将该矩阵的另一维度设置为固定值3，你可以按照自己的实际需求增加相应的代码以适应变化的输入值。
    """
    returnMat = zeros((numberOfLines,3))        #prepare matrix to return
    classLabelVector = []                       #prepare labels return   
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector
"""
在Python命令提示符下输入下面命令：
reload(KNN)
datingDataMat, datingLabels = KNN.file2matrix('datingTestSet.txt')
print(datingDataMat)
"""

"""
分析数据：使用 Matplotlib 创建散点图
"""
import matplotlib
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2])
plt.show()
"""
由于没有使用样本分类的特征值，我们很难从上图中看到任何有用的数据模式信息。
一般来说，我们会采用色彩或其他的记号来标记不同样本分类，以便更好地理解数据信息。
Matplotlib库提供的scatter函数支持个性化标记散点图上的点。
"""
ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0 * array(datingLabels), 15.0 * array(datingLabels))

"""
准备数据：归一化数值
    我们很容易发现，根据欧式距离公式数字差值最大的属性对计算结果的影响最大
    在处理这种不同取值范围的特征值时，我们通常采用的方法是将数值归一化，如将取值范围处理为0到1或者-1到1之间。
    下面的公式可以将任意取值范围的特征值转化为0到1区间内的值：
        newValue = (oldValue - min) / (max - min)
    其中min和max分别是数据集中的最小特征值和最大特征值。虽然改变数值取值范围增加了分类器的复杂度，但为了得到准确结果，我们必须这样做。
    增加一个新函数autoNorm()，该函数可以自动将数字特征值转化为0到1的区间。
"""
def autoNorm(dataSet):
    """
    我们使用NumPy库中tile()函数将变量内容复制成输入矩阵同样大小的矩阵，
        注意这是具体特征值相除，而对于某些数值处理软件包，/可能意味着矩阵除法，但在NumPy库中，矩阵除法需要使用函数 linalg.solve(matA,matB)。
    """
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))   #element wise divide
    return normDataSet, ranges, minVals

"""
在Python命令提示符下输入下面命令：
reload(KNN)
normMat, ranges, minVals = KNN.autoNorm(datingDataMat)
print(normMat)
print(ranges)
print(minVals)
"""

"""
测试算法：作为完整程序验证分类器
    机器学习算法一个很重要的工作就是评估算法的正确率，通常我们只提供已有数据的90%作为训练样本来训练分类器，而使用其余的10%数据去测试分类器，检测分类器的正确率。
    需要注意的是，10%的测试数据应该是随机选择的，由于海伦提供的数据并没有按照特定目的来排序，所以我们可以随意选择10%数据而不影响其随机性。

    前面我们已经提到可以使用错误率来检测分类器的性能。
        对于分类器来说，错误率就是分类器给出错误结果的次数除以测试数据的总数，完美分类器的错误率为0，而错误率为1.0的分类器不会给出任何正确的分类结果。
"""
def datingClassTest():
    hoRatio = 0.50      #hold out 10%
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')       #load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))
    print(errorCount)

"""
在Python命令提示符下重新加载kNN模块，并输入kNN.datingClassTest()，执行分类器测试程序
我们可以改变函数datingClassTest内变量hoRatio和变量k的值，检测错误率是否随着变量值的变化而增加。
"""

"""
使用算法：构建完整可用系统
    上面我们已经在数据上对分类器进行了测试，现在终于可以使用这个分类器为海伦来对人们分类。
    我们会给海伦一小段程序，通过该程序海伦会在约会网站上找到某个人并输入他的信息。
    程序会给出她对对方喜欢程度的预测值。
"""
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses'] 
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?")) 
    iceCream = float(input("liters of ice cream consumed per year?")) 
    datingDataMat, datingLabels = file2matrix ('datingTestSet2.txt') 
    normMat, ranges, minVals = autoNorm (datingDataMat)
    inArr = array ([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals)/ ranges, normMat, datingLabels, 3)
    print("You will probably like this person: ", resultList[classifierResult - 1])

"""
示例：手写识别系统

    本节我们一步步地构造使用k-近邻分类器的手写识别系统。
    为了简单起见，这里构造的系统只能识别数字0到9。
    需要识别的数字已经使用图形处理软件，处理成具有相同的色彩和大小：宽高是32像素×32像素的黑白图像。
    尽管采用文本格式存储图像不能有效地利用内存空间，但是为了方便理解，我们还是将图像转换为文本格式。

    (1) 收集数据：提供文本文件。
    (2) 准备数据：编写函数classify0()，将图像格式转换为分类器使用的list格式。
    (3) 分析数据：在Python命令提示符中检查数据，确保它符合要求
    (4) 训练算法：此步骤不适用于k-近邻算法。
    (5) 测试算法：编写函数使用提供的部分数据集作为测试样本，测试样本与非测试样本的区别在于测试样本是已经完成分类的数据，如果预测分类与实际类别不同，则标记为一个错误。
    (6) 使用算法：本例没有完成此步骤，若你感兴趣可以构建完整的应用程序，从图像中提取数字，并完成数字识别，美国的邮件分拣系统就是一个实际运行的类似系统。
"""

"""
准备数据：将图像转换为测试向量
    我们使用目录trainingDigits中的数据训练分类器，使用目录testDigits中的数据测试分类器的效果。
    两组数据没有重叠，你可以检查一下这些文件夹的文件是否符合要求。

    为了使用前面两个例子的分类器，我们必须将图像格式化处理为一个向量。
    我们将把一个32×32的二进制图像矩阵转换为1×1024的向量，这样前两节使用的分类器就可以处理数字图像信息了。

    首先编写一段函数img2vector，将图像转换为向量
"""
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

"""
testVector = KNN.img2vector('testDigits/0_13.txt')
testVector[0, 0:31]
testVector[0, 32:63]
"""

"""
测试算法：使用 k-近邻算法识别手写数字
"""
from os import listdir

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')           #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): 
            errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount) 
    print("\nthe total error rate is: %f" % (errorCount/float(mTest))) 

"""
在Python命令提示符中输入kNN.handwritingClassTest()，测试该函数的输出结果。

k-近邻算法识别手写数字数据集，错误率为1.2%。改变变量k的值、修改函数handwritingClassTest随机选取训练样本、改变训练样本的数目，都会对k-近邻算法的错误率产生影响，感兴趣的话可以改变这些变量值，观察错误率的变化。

实际使用这个算法时，算法的执行效率并不高。
因为算法需要为每个测试向量做2000次距离计算，每个距离计算包括了1024个维度浮点运算，总计要执行900次，
此外，我们还需要为测试 向量准备2MB的存储空间。

是否存在一种算法减少存储空间和计算时间的开销呢？k决策树就是k-近邻算法的优化版，可以节省大量的计算开销。

本章小结

k-近邻算法是分类数据最简单最有效的算法，本章通过两个例子讲述了如何使用k-近邻算法构造分类器。
k-近邻算法是基于实例的学习，使用算法时我们必须有接近实际数据的训练样本数据。
k-近邻算法必须保存全部数据集，如果训练数据集的很大，必须使用大量的存储空间。
此外，由于必须对数据集中的每个数据计算距离值，实际使用时可能非常耗时。

k-近邻算法的另一个缺陷是它无法给出任何数据的基础结构信息，因此我们也无法知晓平均实例样本和典型实例样本具有什么特征。
下一章我们将使用概率测量方法处理分类问题，该算法可以解决这个问题。
"""
