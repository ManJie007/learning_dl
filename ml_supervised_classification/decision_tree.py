"""
|----决策树的构造
|----在 Python 中使用 Matplotlib 注解绘制树形图
|----测试和存储分类器
|----小结


你是否玩过二十个问题的游戏，游戏的规则很简单：
    参与游戏的一方在脑海里想某个事物，其他参与者向他提问题，只允许提20个问题，问题的答案也只能用对或错回答。
    问问题的人通过推断分解，逐步缩小待猜测事物的范围。

决策树的工作原理与20个问题类似，用户输入一系列数据，然后给出游戏的答案。

近来的调查表明决策树也是最经常使用的数据挖掘算法。

下面流程图就是一个决策树，
    question代表判断模块（decision block），answer代表终止模块（terminating block），表示已经得出结论，可以终止运行。
    从判断模块引出的左右箭头称作分支（branch），它可以到达另一个判断模块或者终止模块

                ------------
                | question |
                ------------
              y /           \  n 
               /             \
        ----------           ------------
        | answer |           | question |
        ----------           ------------
                          y /           \ n
                           /             \
                    ----------           ----------
                    | answer |           | answer |
                    ----------           ----------

k-近邻算法可以完成很多分类任务，但是它最大的缺点就是无法给出数据的内在含义，决策树的主要优势就在于数据形式非常容易理解。

本章构造的决策树算法能够读取数据集合，构建类似上图的决策树。

决策树的一个重要任务是为了数据中所蕴含的知识信息，因此决策树可以使用不熟悉的数据集合，并从中提取出一系列规则，在这些机器根据数据集创建规则时，就是机器学习的过程。

专家系统中经常使用决策树，而且决策树给出结果往往可以匹敌在当前领域具有几十年工作经验的人类专家。

现在我们已经大致了解了决策树可以完成哪些任务，接下来我们将学习如何从一堆原始数据中构造决策树。

首先我们讨论构造决策树的方法，以及如何编写构造树的Python代码；
接着提出一些度量算法成功率的方法；
最后使用递归建立分类器，并且使用Matplotlib绘制决策树图。
构造完成决策树分类器之后，我们将输入一些隐形眼镜的处方数据，并由决策树分类器预测需要的镜片类型。

优点：计算复杂度不高，输出结果易于理解，对中间值的缺失不敏感，可以处理不相关特征数据。
缺点：可能会产生过度匹配问题。
适用数据类型：数值型和标称型。

决策树的一般流程
    (1) 收集数据：可以使用任何方法。
    (2) 准备数据：树构造算法只适用于标称型数据，因此数值型数据必须离散化。
    (3) 分析数据：可以使用任何方法，构造树完成之后，我们应该检查图形是否符合预期。
    (4) 训练算法：构造树的数据结构。
    (5) 测试算法：使用经验树计算错误率。
    (6) 使用算法：此步骤可以适用于任何监督学习算法，而使用决策树可以更好地理解数据的内在含义。

1 决策树的构造
    首先我们讨论数学上如何使用信息论划分数据集，
    然后编写代码将理论应用到具体的数据集上，
    最后编写代码构建决策树。

    在构造决策树时，
        我们需要解决的第一个问题就是，当前数据集上哪个特征在划分数据分类时起决定性作用。
        为了找到决定性的特征，划分出最好的结果，我们必须评估每个特征。
        完成测试之后，原始数据集就被划分为几个数据子集。
        这些数据子集会分布在第一个决策点的所有分支上。
            如果某个分支下的数据属于同一类型，则当前无需阅读的垃圾邮件已经正确地划分数据分类，无需进一步对数据集进行分割。
            如果数据子集内的数据不属于同一类型，则需要重复划分数据子集的过程。
            如何划分数据子集的算法和划分原始数据集的方法相同，直到所有具有相同类型的数据均在一个数据子集内。

        创建分支的伪代码函数createBranch()如下所示：
            检测数据集中的每个子项是否属于同一分类： 
                If so return 类标签；
                Else 
                    寻找划分数据集的最好特征
                    划分数据集
                    创建分支节点
                    for 每个划分的子集
                        调用函数createBranch并增加返回结果到分支节点中
                    return 分支节点

        上面的伪代码createBranch是一个递归函数，在倒数第二行直接调用了它自己。
        后面我们将把上面的伪代码转换为Python代码，这里我们需要进一步了解算法是如何划分数据集的。

    一些决策树算法采用二分法划分数据，本书并不采用这种方法。
    如果依据某个属性划分数据将会产生4个可能的值，我们将把数据划分成四块，并创建四个不同的分支。
    本书将使用ID3算法划分数据集，该算法处理如何划分数据集，何时停止划分数据集（进一步的信息可以参见 http://en.wikipedia.org/wiki/ID3_algorithm）。
    每次划分数据集时我们只选取一个特征属性，如果训练集中存在20个特征，第一次我们选择哪个特征作为划分的参考属性呢？
    我们可以使用多种方法划分数据集，但是每种方法都有各自的优缺点。
    划分数据集的大原则是：将无序的数据变得更加有序。
        组织杂乱无章数据的一种方法就是使用信息论度量信息，信息论是量化处理信息的分支科学。
        我们可以在划分数据之前或之后使用信息论量化度量信息的内容。 
        信息增益
            在划分数据集之前之后信息发生的变化称为信息增益，知道如何计算信息增益，我们就可以计算每个特征值划分数据集获得的信息增益，获得信息增益最高的特征就是最好的选择。
            在可以评测哪种数据划分方式是最好的数据划分之前，我们必须学习如何计算信息增益。
        集合信息的度量方式称为香农熵或者简称为熵，这个名字来源于信息论之父克劳德·香农。
            熵定义为信息的期望值，在明晰这个概念之前，我们必须知道信息的定义。
            如果待分类的事务可能划分在多个分类之中，则符号xi的信息定义为
                
                l(xi) = -log_{2}p(xi)

            其中p(xi)是选择该分类的概率。
            为了计算熵，我们需要计算所有类别所有可能值包含的信息期望值，通过下面的公式得到：

                     n
                H = -Σ p(xi)log_{2}p(xi)
                    i=1

            其中n是分类的数目。
"""

#学习如何使用Python计算信息熵
from math import log

def calcshannonent(dataset):
    numentries = len(dataset)
    labelcounts = {}
    for featvec in dataset: #the the number of unique elements and their occurance
        #假设数据集的最后一列是标签（类别），使用 featvec[-1] 获取当前样本的标签
        currentlabel = featvec[-1]
        if currentlabel not in labelcounts.keys(): labelcounts[currentlabel] = 0
        labelcounts[currentlabel] += 1
    shannonent = 0.0
    for key in labelcounts:
        prob = float(labelcounts[key])/numentries
        shannonent -= prob * log(prob,2) #log base 2
    return shannonent

def createDataSet():
    """
    可以利用createDataSet()函数得到的简单鱼鉴定数据集，你可以输入自己的createDataSet()函数
    """
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

"""
在Python命令提示符下输入
reload(decision_tree.py.py)
myDat, labels = decision_tree.createDataSet()
print(myDat)
decision_tree.calcshannonent(myDat)

熵越高，则混合的数据也越多，我们可以在数据集中添加更多的分类，观察熵是如何变化的。

这里我们增加第三个名为maybe的分类，测试熵的变化:
myDat[0][-1]='maybe'
print(myDat)
decision_tree.calcshannonent(myDat)

熵变大了...

    得到熵之后，我们就可以按照获取最大信息增益的方法划分数据集，下一节我们将具体学习如何划分数据集以及如何度量信息增益。

    另一个度量集合无序程度的方法是基尼不纯度（Gini impurity），简单地说就是从一个数据集中随机选取子项，度量其被错误分类到其他分组里的概率。
    本书不采用基尼不纯度方法，这里就不再做进一步的介绍。下面我们将学习如何划分数据集，并创建决策树。

2.划分数据集
    上节我们学习了如何度量数据集的无序程度，分类算法除了需要测量信息熵，还需要划分数据集，度量划分数据集的熵，以便判断当前是否正确地划分了数据集。
    我们将对每个特征划分数据集的结果计算一次信息熵，然后判断按照哪个特征划分数据集是最好的划分方式。
    想象一个分布在二维空间的数据散点图，需要在数据之间划条线，将它们分成两部分，我们应该按照x轴还是y轴划线呢？答案就是本节讲述的内容。
"""
def splitDataSet(dataSet, axis, value):
    """
    根据给定的特征轴（axis）和特征值（value）从原始数据集中筛选出符合条件的子集。
    """
    """
    需要注意的是，Python语言不用考虑内存分配问题。
    Python语言在函数中传递的是列表的引用，在函数内部对列表对象的修改，将会影响该列表对象的整个生存周期。

    为了消除这个不良影响，我们需要在函数的开始声明一个新列表对象。
    因为该函数代码在同一数据集上被调用多次，为了不修改原始数据集，创建一个新的列表对象 。
    """
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]     #chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

"""
reload(decision_tree.py.py)
myDat, labels = decision_tree.createDataSet()
print(myDat)
decision_tree.splitDataSet(myDat, 0, 1)
decision_tree.splitDataSet(myDat, 0, 0)
"""

"""
    接下来我们将遍历整个数据集，循环计算香农熵和splitDataSet()函数，找到最好的特征划分方式。
    熵计算将会告诉我们如何划分数据集是最好的数据组织方式。
"""
def chooseBestFeatureToSplit(dataSet):
    """
    在函数中调用的数据需要满足一定的要求：
    第一个要求是，数据必须是一种由列表元素组成的列表，而且所有的列表元素都要具有相同的数据长度；
    第二个要求是，数据的最后一列或者每个实例的最后一个元素是当前实例的类别标签。

    数据集一旦满足上述要求，我们就可以在函数的第一行判定当前数据集包含多少特征属性。
    我们无需限定list中的数据类型，它们既可以是数字也可以是字符串，并不影响实际计算。
    """
    numFeatures = len(dataSet[0]) - 1      #the last column is used for the labels
    baseEntropy = calcshannonent(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):        #iterate over all the features
        featList = [example[i] for example in dataSet]#create a list of all the examples of this feature
        uniqueVals = set(featList)       #get a set of unique values
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcshannonent(subDataSet)     
        infoGain = baseEntropy - newEntropy     #calculate the info gain; ie reduction in entropy
        if (infoGain > bestInfoGain):       #compare this to the best gain so far
            bestInfoGain = infoGain         #if better than current best, set to best
            bestFeature = i
    return bestFeature                      #returns an integer

"""
reload(decision_tree.py.py)
myDat, labels = decision_tree.createDataSet()
chooseBestFeatureToSplit(myDat) #0

代码运行结果告诉我们，第0个特征是最好的用于划分数据集的特征。
结果是否正确呢？这个结果又有什么实际意义呢？

    数据集中的数据
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    如果我们按照第一个特征属性划分数据，也就是说第一个特征是1的放在一个组，第一个特征是0的放在另一个组，
    数据一致性如何？
        按照上述的方法划分数据集，第一个特征为1的海洋生物分组将有两个属于鱼类，一个属于非鱼类；另一个分组则全部属于非鱼类。
    如果按照第二个特征分组，结果又是怎么样呢？
        第一个海洋动物分组将有两个属于鱼类，两个属于非鱼类；另一个分组则只有一个非鱼类。
    第一种划分很好地处理了相关数据。

    本节我们学习了如何度量数据集的信息熵，如何有效地划分数据集，下一节我们将介绍如何将这些函数功能放在一起，构建决策树。

3.递归构建决策树

    目前我们已经学习了从数据集构造决策树算法所需要的子功能模块，其工作原理如下：
        得到原始数据集，然后基于最好的属性值划分数据集，由于特征值可能多于两个，因此可能存在大于两个分支的数据集划分。
        第一次划分之后，数据将被向下传递到树分支的下一个节点，在这个节点上，我们可以再次划分数据。
        因此我们可以采用递归的原则处理数据集。

        递归结束的条件是：程序遍历完所有划分数据集的属性，或者每个分支下的所有实例都具有相同的分类。
        如果所有实例具有相同的分类，则得到一个叶子节点或者终止块。
        任何到达叶子节点的数据必然属于叶子节点的分类，参见图3-2所示。

                                No surfacing    Flippers?     Fish?
                             1.         Yes         Yes       Yes
                             2.         Yes         Yes       Yes
                             3.         Yes          No        No
                             4.          No         Yes        No
                             5.          No         Yes        No
                                
                                -----------------
                                | No Surfacing? |
                                -----------------
                                /               \
                           No  /                 \ Yes
                              /                   \                        Flippers?     Fish?         
                        ------                   -------------          1.     Yes       Yes
                        | No |                   | Flippers? |          2.     Yes       Yes
                        ------                   -------------          3.      No        No
                   Flippers?     Fish?           /            \    
                4.     Yes        No         No /              \  Yes
                5.     Yes        No           /                \
                                         ------                  -------
                                 Fish?   | No |                  | Yes |        Fish?
                              3.  No     ------                  -------     4. Yes
                                                                             5. Yes

        第一个结束条件使得算法可以终止，我们甚至可以设置算法可以划分的最大分组数目。
        后续章节还会介绍其他决策树算法，如C4.5和CART，这些算法在运行时并不总是在每次划分分组时都会消耗特征。
        由于特征数目并不是在每次划分数据分组时都减少，因此这些算法在实际使用时可能引起一定的问题。
        目前我们并不需要考虑这个问题，只需要在算法开始运行前计算列的数目，查看算法是否使用了所有属性即可。
        如果数据集已经处理了所有属性，但是类标签依然不是唯一的，此时我们需要决定如何定义该叶子节点，在这种情况下，我们通常会采用多数表决的方法决定该叶子节点的分类。
"""
import operator

def majorityCnt(classList):
    """
    该函数使用分类名称的列表，
    然后创建键值为classList中唯一值的数据字典，
    字典对象存储了classList中每个类标签出现的频率，
    最后利用operator操作键值排序字典，
    并返回出现次数最多的分类名称
    """
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


#创建树的函数代码
def createTree(dataSet,labels):
    """
    Args:
        数据集和标签列表。
        标签列表包含了数据集中所有特征的标签，算法本身并不需要这个变量，但是为了给出数据明确的含义，我们将它作为一个输入参数提供。
        此外，前面提到的对数据集的要求这里依然需要满足。
            在函数中调用的数据需要满足一定的要求：
            第一个要求是，数据必须是一种由列表元素组成的列表，而且所有的列表元素都要具有相同的数据长度；
            第二个要求是，数据的最后一列或者每个实例的最后一个元素是当前实例的类别标签。
    """
    #debug
    import pdb
    pdb.set_trace()
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList): 
        return classList[0]#stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1: #stop splitting when there are no more features in dataSet
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        """
        这行代码复制了类标签，并将其存储在新列表变量subLabels中。
        之所以这样做，是因为在Python语言中函数参数是列表类型时，参数是按照引用方式传递的。
        为了保证每次调用函数createTree()时不改变原始列表的内容，使用新变量subLabels代替原始列表。
        """
        subLabels = labels[:]       #copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree                            

"""
在Python命令提示符下输入下列命令
reload(trees)
myDat, labels = trees.decision_tree.createDataSet()
myTree = trees.createTree(myDat, labels)
print(myTree)
{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}

变量myTree包含了很多代表树结构信息的嵌套字典。
如果值是类标签，则该子节点是叶子节点；如果值是另一个数据字典，则子节点是一个判断节点，这种格式结构不断重复就构成了整棵树。
"""

"""
在 Python 中使用 Matplotlib 注解绘制树形图

决策树的主要优点就是直观易于理解，如果不能将其直观地显示出来，就无法发挥其优势。
|----Matplotlib 注解
|       Matplotlib提供了一个注解工具annotations，可以在数据图形上添加文本注释。
|       注解通常用于解释数据的内容。
|       由于数据上面直接存在文本描述非常丑陋，因此工具内嵌支持带箭头的划线工具，使得我们可以在其他恰当的地方指向数据位置，并在此处添加描述信息，解释数据内容。
|----构造注解树
        我们虽然有x、y坐标，但是如何放置所有的树节点却是个问题。
        我们必须知道有多少个叶节点，以便可以正确确定x轴的长度；我们还需要知道树有多少层，以便可以正确确定y轴的高度。
        定义两个新函数getNumLeafs()和getTreeDepth()，来获取叶节点的数目和树的层数
"""
import matplotlib.pyplot as plt

#定义文本框和箭头格式
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

#绘制带箭头的注解
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    """
    Args:
        nodeTxt：节点内显示的文本。
        centerPt：节点中心的坐标。
        parentPt：父节点的坐标，用于连接箭头。
        nodeType：节点样式（如 decisionNode 或 leafNode）。
    """
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
             xytext=centerPt, textcoords='axes fraction',
             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args )

def createPlot():
    """
    首先创建了一个新图形并清空绘图区，然后在绘图区上绘制两个代表不同类型的树节点，后面我们将用这两个节点绘制树形图。
    """
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses 
    plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()

"""
测试

import treePlotter
treePlotter.createPlot()
"""

def getNumLeafs(myTree):
    numLeafs = 0
    """
    第一个关键字是第一次划分数据集的类别标签，附带的数值表示子节点的取值。
    从第一个关键字出发，我们可以遍历整棵树的所有子节点。
    """
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes
            """
            如果子节点是字典类型，则该节点也是一个判断节点，需要递归调用getNumLeafs()函数。
            getNumLeafs()函数遍历整棵树，累计叶子节点的个数，并返回该数值。
            """
            numLeafs += getNumLeafs(secondDict[key])
        else:   numLeafs +=1
    return numLeafs

def getTreeDepth(myTree):
    """
    第2个函数getTreeDepth()计算遍历过程中遇到判断节点的个数。该函数的终止条件是叶子节点，一旦到达叶子节点，则从递归调用中返回，并将计算树深度的变量加一。
    """
    maxDepth = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:   thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth

def retrieveTree(i):
    """
    为了节省大家的时间，函数retrieveTree输出预先存储的树信息，避免了每次测试代码时都要从数据中创建树的麻烦。
    """
    listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                  {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                  ]
    return listOfTrees[i]

"""
测试
reload(treePlotter)
treePlotter.retrieveTree(1)
myTree = treePlotter.retrieveTree(0)
treePlotter.getNumLeafs(myTree)
treePlotter.getTreeDepth(myTree)
"""

def plotMidText(cntrPt, parentPt, txtString):
    """
    在父子节点间填充文本信息
    """
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

def plotTree(myTree, parentPt, nodeTxt):#if the first key tells you what feat was split on
    """
    首先计算树的宽和高
    树的宽度用于计算放置判断节点的位置，主要的计算原则是将它放在所有叶子节点的中间，而不仅仅是它子节点的中间。
    同时我们使用两个全局变量plotTree.xOff和plotTree.yOff追踪已经绘制的节点位置，以及放置下一个节点的恰当位置。
    另一个需要说明的问题是，绘制图形的x轴有效范围是0.0到1.0，y轴有效范围也是0.0～1.0。
    通过计算树包含的所有叶子节点数，划分图形的宽度，从而计算得到当前节点的中心位置，也就是说，我们按照叶子节点的数目将x轴划分为若干部分。
        按照图形比例绘制树形图的最大好处是无需关心实际输出图形的大小，一旦图形大小发生了变化，函数会自动按照图形大小重新绘制。
    """
    numLeafs = getNumLeafs(myTree)  #this determines the x width of this tree
    depth = getTreeDepth(myTree)
    firstStr = myTree.keys()[0]     #the text label for this node should be this
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)

    """绘出子节点具有的特征值，或者沿此分支向下的数据实例必须具有的特征值"""
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]

    """
    按比例减少全局变量plotTree.yOff，并标注此处将要绘制子节点
    这些节点既可以是叶子节点也可以是判断节点，此处需要只保存绘制图形的轨迹。
    因为我们是自顶向下绘制图形，因此需要依次递减y坐标值，而不是递增y坐标值。
    """
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes   
            plotTree(secondDict[key],cntrPt,str(key))        #recursion
        else:   #it's a leaf node print the leaf node
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    """
    在绘制了所有子节点之后，增加全局变量Y的偏移。
    每完成一个分支的绘制后，y 偏移量恢复上一层，确保其他分支的绘制不会受影响。
    """
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
#if you do get a dictonary you know it's a tree, and the first element will be another dict

def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)    #no ticks
    #createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses 
    """
    全局变量plotTree.totalW存储树的宽度，全局变量plotTree.totalD存储树的深度，我们使用这两个变量计算树节点的摆放位置，这样可以将树绘制在水平方向和垂直方向的中心位置。
    """
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;
    plotTree(inTree, (0.5,1.0), '')
    plt.show()

"""
测试
reload(treePlotter)
myTree = treePlotter.retrieveTree(0)
treePlotter.createPlot(myTree)
myTree['no surfacing']['3']='mabye'
treePlotter.createPlot(myTree)
"""

"""
测试和存储分类器
|----将使用决策树构建分类器，以及实际应用中如何存储分类器。
|       依靠训练数据构造了决策树之后，我们可以将它用于实际数据的分类。
|       在执行数据分类时，需要决策树以及用于构造树的标签向量。然后，程序比较测试数据与决策树上的数值，递归执行该过程直到进入叶子节点；
|       最后将测试数据定义为叶子节点所属的类型。

|       如何在硬盘上存储决策树分类器
|----在真实数据上使用决策树分类算法，验证它是否可以正确预测出患者应该使用的隐形眼镜类型。
"""
def classify(inputTree,featLabels,testVec):
    """
    在存储带有特征的数据会面临一个问题：
        程序无法确定特征在数据集中的位置，例如前面例子的第一个用于划分数据集的特征是no surfacing属性，但是在实际数据集中该属性存储在哪个位置？是第一个属性还是第二个属性？
        特征标签列表将帮助程序处理这个问题。使用index方法查找当前列表中第一个匹配firstStr变量的元素 。
        然后代码递归遍历整棵树，比较testVec变量中的值与树节点的值，如果到达叶子节点，则返回当前节点的分类标签。
    """
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    #将标签字符串转换为索引
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel

"""
myDat, labels=trees.createDataSet()
print(labels)
myTree = treePlotter.retrieveTree(0)
print(myTree)
trees.classify(myTree, labels, [1, 0])
trees.classify(myTree, labels, [1, 1])
"""

"""
决策树的存储
    构造决策树是很耗时的任务
    然而用创建好的决策树解决分类问题，则可以很快完成。
        因此，为了节省计算时间，最好能够在每次执行分类时调用已经构造好的决策树。
        为了解决这个问题，需要使用Python模块pickle序列化对象。
        序列化对象可以在磁盘上保存对象，并在需要的时候读取出来。
        任何对象都可以执行序列化操作，字典对象也不例外。
"""
def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()
    
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)

"""
测试
reload(trees)
trees.storeTree(myTree, 'classifierStorage.txt')
trees.grabTree('classifierStorage.txt')


我们可以将分类器存储在硬盘上，而不用每次对数据分类时重新学习一遍，这也是决策树的优点之一，像第2章介绍了k-近邻算法就无法持久化分类器。

我们可以预先提炼并存储数据集中包含的知识信息，在需要对事物进行分类时再使用这些知识。
"""

"""
示例：使用决策树预测隐形眼镜类型
    
    (1) 收集数据：提供的文本文件。 
    (2) 准备数据：解析tab键分隔的数据行。 
    (3) 分析数据：快速检查数据，确保正确地解析数据内容，使用createPlot()函数绘制最终的树形图。
    (4) 训练算法：使用createTree()函数。 
    (5) 测试算法：编写测试函数验证决策树可以正确分类给定的数据实例。 
    (6) 使用算法：存储树的数据结构，以便下次使用时无需重新构造树。 

隐形眼镜数据集
    The dataset is a modified version of the Lenses dataset retrieved from the UCI Machine Learning Repository November 3, 2010 [http://archive.ics.uci.edu/ml/machine-learning-databases/lenses/]. 
    The source of the data is Jadzia Cendrowska and was originally published in “PRISM: An algorithm for inducing modular rules,” in International Journal of Man-Machine Studies (1987), 27, 349–70.)

>>>fr=open('lenses.txt')
>>>lenses=[inst.strip().split('\t') for inst in fr.readlines()]
>>>lensesLabels=['age','prescript','astigmatic','tearRate']
>>>lensesTree = trees.createTree(lenses,lensesLabels)
>>>lensesTree
{'tearRate':{'reduced':'no lenaes','normal':{'astigmatic':{'yes':{'prescript':{'hyper':{'age':{'pre':'no lenses','presbyopic':'no lenses','young':'hard'}},'myope':'hard'}},'no':{'age':{'pre':'soft','presbyopic':{'prescript':{'hyper':'soft','myope':'no lenses'}},'young':'soft'}}}}}}
>>>treePlotter.createPlot(lensesTree)

所示的决策树非常好地匹配了实验数据，然而这些匹配选项可能太多了。
我们将这种问题称之为过度匹配（overfitting）。

为了减少过度匹配问题，我们可以裁剪决策树，去掉一些不必要的叶子节点。
如果叶子节点只能增加少许信息，则可以删除该节点，将它并入到其他叶子节点中。
第9章将进一步讨论这个问题。 
    第9章将学习另一个决策树构造算法CART，本章使用的算法称为ID3，它是一个好的算法但并不完美。
    ID3算法无法直接处理数值型数据，尽管我们可以通过量化的方法将数值型数据转化为标称型数值，但是如果存在太多的特征划分，ID3算法仍然会面临其他问题。
"""

"""
决策树分类器就像带有终止块的流程图，终止块表示分类结果。
开始处理数据集时，我们首先需要测量集合中数据的不一致性，也就是熵，然后寻找最优方案划分数据集，直到数据集中的所有数据属于同一分类。
    ID3算法可以用于划分标称型数据集。
构建决策树时，我们通常采用递归的方法将数据集转化为决策树。
    一般我们并不构造新的数据结构，而是使用Python语言内嵌的数据结构字典存储树节点信息。

使用Matplotlib的注解功能，我们可以将存储的树结构转化为容易理解的图形。
Python语言的pickle模块可用于存储决策树的结构。

隐形眼镜的例子表明决策树可能会产生过多的数据集划分，从而产生过度匹配数据集的问题。
我们可以通过裁剪决策树，合并相邻的无法产生大量信息增益的叶节点，消除过度匹配问题。

还有其他的决策树的构造算法，最流行的是C4.5和CART，第9章讨论回归问题时将介绍CART算法。

KNN、decision_tree讨论的是结果确定的分类算法，数据实例最终会被明确划分到某个分类中。
"""
