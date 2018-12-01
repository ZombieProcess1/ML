'''
朴素贝叶斯算法用来进行文本分类、邮件归类实验的一般步骤
1、文本分词，需要按照空格划分单词，去掉空格、符号、长度太短单词、停用词（辅助结构词汇）、高频词
2、提取出词汇集合，并能够对每条文本用向量表示，按照词汇列表顺序，该词出现则用1表示，没有则用0表示
3、测试集、训练集随机划分，训练相应的贝叶斯分类器
4、针对需要测试的样例，进行分类
在后续实验进行之前，我将先用简单的代码复现一下《统计学习方法》中的例4.1，然后在后面指出与《机器学习实战》中文本分析的略微不同
'''
'''
《统计学习方法》例4.1：试由下表学习一个朴素贝叶斯分类器并确定x=(2,'S')的类标记。
样例号  1  2  3  4  5  6  7  8  9  10  11  12  13  14  15
属性X1  1  1  1  1  1  2  2  2  2  2   3   3   3   3   3
属性X2  S  M  M  S  S  S  M  M  L  L   L   M   M   L   L
分类Y  -1 -1  1  1 -1 -1 -1  1  1  1   1   1   1   1  -1
每一列数据代表一个样本，最后一行代表样本的分类，且分类集合是（-1,1）
trainList =[[1,'S'],[1,'M'],[1,'M'],[1,'S'],[1,'S'],[2,'S'],[2,'M'],[2,'M'],
                [2,'L'],[2,'L'],[3,'L'],[3,'M'],[3,'M'],[3,'L'],[3,'L']]
trainLabes = [-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1]

根据贝叶斯的思想：需要求解的是P(c)*p(x|c)，而x在分类前提下是可以得到每个属性出现的概率，又各个概率独立所以想乘
'''
from numpy import *
class Bayes1():
    def __init__(self):
        pass
    def tempListInit(self,trainLists,entryNum,m):
        '''
        该函数用于解释bayesClassify()函数中的注解1
        在《统计学习方法》针对贝叶斯估计提出了可能所要估计的概率值为0的情况，也就是在某个分类下，某一属性值出现的概率就是0，那么最终测试的概率也是零，这种情况显然需要避免
        原本贝叶斯估计是P(Y=C) = number(Y=C)/number(样本总数)
        p(Xi=a|Y=C) = number(Xi=a & Y=C)/number(Y=C)
        考虑到概率为0的情况，所以将公式改为
        P(Y=C) = (number(Y=C)+Q)/(number(样本总数)+Q*number(样本种类数))  Q是给定的一个参数，一般定为1
        P(Xi=a\Y=C) = (number(Xi=a & Y=C)+Q)/(number(Y=C)+Q*number(该属性的值数量))
        可以发现其实这两个式子的格式一样，只不过前者分母是样本总数加上参数乘以分类数量
        后者的分母是该分类下样本总数加上参数乘以属性值数量
        trainLists:训练样本集
        entryNum:属性列数
        m:分类标签数量
        :return:定义的每个分类下，与输入相同属性的向量和，以及样本总数
        '''
        entryListNumber = []  #用于存储每列属性值的数量
        tempList = [] #return值
        for i in range(entryNum):
            entrylistsI = [example[i] for example in trainLists]
            entrylistsI = set(entrylistsI)
            entryListNumber.append(len(entrylistsI))
        for j in range(m):
            tempList.append([[1]*entryNum,entryListNumber])
        return tempList
    def bayesClassify(self,trainList,trainLabels,inputTest): #inputTest (2,'S')
        labelList = list(set(trainLabels))  # 训练集总共分类标签集合
        m = len(labelList)
        entryNum = len(trainList[0]) #属性列数
        #tempList = [[[0]*entryNum,0]]*m  #这种写法不可取，会在注解2的位置同步加
        tempList = []   #该列表每个元素用于存储一个分类对应的各个向量之和，以及各分类的总数
        for t in range(m):tempList.append([[0]*entryNum,0])  #注解1
        for index1 in range(len(trainList)):
            label = trainLabels[index1]
            tempIndex = labelList.index(label)
            for index2 in range(entryNum):
                if trainList[index1][index2] == inputTest[index2] :
                    tempList[tempIndex][0][index2] += 1  #注解2：该样例属性值与测试样例属性相等，则向量对应属性数量增加
            tempList[tempIndex][1] += 1   #该种类总数增加
        tempMax  = 0
        inputLabel = labelList[0]
        for index3 in range(m):
            tempVec = array(tempList[index3][0])/float(tempList[index3][1])
            tempVecCheng = 1
            for i in tempVec:
                tempVecCheng  *= i
            tempP = tempVecCheng * tempList[index3][1]/len(trainList)
            print(labelList[index3],"对应的概率是：",tempP)
            if tempP > tempMax:
                tempMax = tempP
                inputLabel = labelList[index3]
        print(inputLabel)
    '''根据注解一改进后的贝叶斯分类函数'''
    def bayesClassify_(self,trainList,trainLabels,inputTest): #inputTest (2,'S')
        labelList = list(set(trainLabels))  # 训练集总共分类标签集合
        m = len(labelList)   #分类总数
        entryNum = len(trainList[0]) #属性列数
        #tempList = [[[0]*entryNum,0]]*m  #这种写法不可取，会在注解2的位置同步加
        tempList = []   #该列表每个元素用于存储一个分类对应的各个向量之和，以及各分类的总数
        #for t in range(m):tempList.append([[0]*entryNum,0])  #注解1
        tempList = self.tempListInit(trainList,entryNum,m)   #根据注解一********************改进
        tempList2 = [1]*m #单纯储存每个分类的样本总数   根据注解一********************改进
        for index1 in range(len(trainList)):
            label = trainLabels[index1]
            tempIndex = labelList.index(label)
            for index2 in range(entryNum):
                if trainList[index1][index2] == inputTest[index2] :
                    tempList[tempIndex][0][index2] += 1  #注解2：该样例属性值与测试样例属性相等，则向量对应属性数量增加
            tempList[tempIndex][1] = [i+1 for i in tempList[tempIndex][1]]   #该种类总数增加 根据注解一********************改进
            tempList2[tempIndex] += 1   #根据注解一********************改进
            print(tempList)
        tempMax  = 0
        inputLabel = labelList[0]
        for index3 in range(m):
            tempVec = array(tempList[index3][0])/array(tempList[index3][1])  #根据注解一********************改进
            tempVecCheng = 1
            for i in tempVec:
                tempVecCheng  *= i
            tempP = tempVecCheng * (tempList2[index3])/(len(trainList)+m)    #根据注解一********************改进
            print(labelList[index3],"对应的概率是：",tempP)
            if tempP > tempMax:
                tempMax = tempP
                inputLabel = labelList[index3]
        print(inputLabel)

if __name__ == '__main__':
    trainList = [[1, 'S'], [1, 'M'], [1, 'M'], [1, 'S'], [1, 'S'], [2, 'S'], [2, 'M'], [2, 'M'],
                 [2, 'L'], [2, 'L'], [3, 'L'], [3, 'M'], [3, 'M'], [3, 'L'], [3, 'L']]
    trainLables = [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]
    test = Bayes1()
    #test.bayesClassify(trainList,trainLables,[2,'S'])
    test.bayesClassify_(trainList,trainLables,[2,'S'])
