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
from math import *
import codecs
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
            #print(tempList)
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
        return inputLabel
    def Bayes_test(self):
        trainList = [[1, 'S'], [1, 'M'], [1, 'M'], [1, 'S'], [1, 'S'], [2, 'S'], [2, 'M'], [2, 'M'],
                     [2, 'L'], [2, 'L'], [3, 'L'], [3, 'M'], [3, 'M'], [3, 'L'], [3, 'L']]
        trainLables = [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]
        #self.bayesClassify(trainList, trainLables, [2, 'S'])
        label = self.bayesClassify_(trainList, trainLables, [2, 'S'])
        print("实例标签是",label)

'''《机器学习实战》文本分类、垃圾邮件'''
class StaticBayes():
    def __init__(self):
        pass
    def textWord(self,text):#该函数用于把文本划分成单词列表
        '''
        :param filename: 文本文件名，比如邮件一个文本就是一个要分类的单词列表
        :return:返回文本单词列表
        '''
        import re
        textLists = re.split(r'\W*',text)
        textLists = [list.lower() for list in textLists if len(list)>2] #用空格、符号划分文本单词，并且只保留长度大于一定值的字符串
        return textLists
    def vocabList(self,trainLists):
        '''
        :param trainLists:训练数据集，为文本单词列表的组合
        :return: 返回所有符合条件的单词集合，该集合用于构建单词向量
        '''
        vocabLists = set([])
        for li in trainLists:
            vocabLists = vocabLists | set(li)
        return list(vocabLists)
    def textListVec(self,vocabLists,inputTextList):
        '''
        每个单词出现或者不出现分别用1,0表示，那么针对每个文本单词列表，都可以构建一个长度为单词集合的1,0向量列表
        不过在具体的文本分析实践中，一个文本中单词出现并且有多个，其数量就应该加和
        :param vocabLists: 唯一单词集
        :param inputTextList: 文本单词列表
        :return: 用向量表示的文本单词列表
        '''
        inputTextVec = [0] * len(vocabLists)
        for word in inputTextList:
            if word in vocabLists:
                inputTextVec[vocabLists.index(word)] += 1
        return inputTextVec
    def BayesTrain(self,trainLists,trainLabels,vocabLists):
        '''
        对所有训练集和给定的单词集加以训练，得到各个分类下相应单词的条件概率以及各分类的条件概率
        下面的函数不仅仅适用于文本分类、邮件分类，同时也适用于《统计学习方法》里面的实例
        假定分类数量未知，总共是Y类
        :param trainLists: 原始训练集
        :param trainLabels:原始标签分类列表
        :param vocabLists: self.vocabList()得到的单一单词集合
        :return: 各分类概率，各分类下条件概率
        '''
        labelSet = list(set(trainLabels))
        Y = len(labelSet) #得到分类个数，比如邮件分类，垃圾和非垃圾是两类
        tempList = []
        '''
        tempList = [[1,1,1,1,1,1,……],[2,2,2,2,2,2,2,2,……],num]
        文本分类与邮件分类  和上面统计学习方法例4.1不同的一点在于，单词的出现实际上并不完全独立
        所以在计算分类下单词的条件概率时需要考虑原本每个单词的出现概率，所以各个单词向量相加之后，分母的值应该是该分类下所有单词出现的总数，而不是文本个数
        且初始化单词向量每个值默认1，每个条件概率分母初始值为属性值个数，而每个单词只有出现不出现两种，所以都是2 num值是该分类样例总数
        '''
        entryNum = len(vocabLists)  #每个样例值属性列数
        entrySetNum = [2]*entryNum #用于储存每列属性值的数量
        for y in range(Y):
            tempList.append([[1]*entryNum,entrySetNum[:],0])
        for trainIndex in range(len(trainLists)):
            trainlist = trainLists[trainIndex]
            trainlabel = trainLabels[trainIndex]
            labelIndex = labelSet.index(trainlabel)
            tempList[labelIndex][0] = list(array(tempList[labelIndex][0])+array(self.textListVec(vocabLists,trainlist)) )#对应分类
            tempList[labelIndex][1] = [temp+sum(self.textListVec(vocabLists,trainlist)) for temp in tempList[labelIndex][1]]
            tempList[labelIndex][2] += 1
        resultList = []  #用于存放每个分类下条件概率、类别概率的结果
        for labelIndex in range(Y):
            print(labelSet[labelIndex],"对应的结果为",tempList[labelIndex])
            pyVec = array(tempList[labelIndex][0])/array(tempList[labelIndex][1])
            pyResult = [log(py) for py in pyVec] #属性列向量条件概率取对数加和
            pcResult = log(tempList[labelIndex][2]/(len(trainLists)+Y))  #分类概率取对数
            resultList.append([pyResult,pcResult])
        return resultList,labelSet
    def BayesClassify(self,resultList,labelSet,vocabLists,testList):
        '''
        用于给定实例的分类
        :param resultList: self.BayesTrain()的结果
        :param labelSet: 同上，分类标签集合
        :param vocabLists: self.vocabList()的结果
        :param testList: 需要分类的实例
        :return: 返回实例的分类标签
        '''
        label = labelSet[0]
        maxpro = -inf
        testVec = self.textListVec(vocabLists,testList)
        for result in resultList:
            pro = sum(array(result[0]) * array(testVec))+result[1]
            print(labelSet[resultList.index(result)],"的概率是",pro)
            if pro>maxpro:
                maxpro = pro
                label = labelSet[resultList.index(result)]
                #print("新标签",label)
        return label
    def testBayes(self):
        '''
        对《机器学习实战》邮件样本进行交叉验证，得到每次分类的错误率
        :return: 返回错误率
        '''
        docLists = []
        labelLists = []
        fullText = []
        for i in range(1,26):
            wordList = self.textWord(codecs.open('data\\email\\spam\%d.txt'%i,'r',encoding='utf-8',errors='ignore').read())
            docLists.append(wordList)
            labelLists.append(1)
            fullText.extend(wordList)
            wordList2 = self.textWord(codecs.open('data\\email\\ham\%d.txt'%i,'r',encoding='utf-8',errors='ignore').read())
            docLists.append(wordList2)
            labelLists.append(0)
            fullText.extend(wordList2)
        vocabLists = self.vocabList(docLists)
        trainLists = list(arange(50))
        testLists = []
        for i in range(10):
            randIndex = int(random.uniform(0,len(trainLists)))
            testLists.append(randIndex)
            del(trainLists[randIndex])
        trainMat = []
        trainClasses = []
        for index in trainLists:
            trainMat.append(docLists[index])
            trainClasses.append(labelLists[index])
        resultLists,labelSet = self.BayesTrain(trainMat,trainClasses,vocabLists)
        errorNum = 0
        for index2 in testLists:
            labelResult = self.BayesClassify(resultLists,labelSet,vocabLists,docLists[index2])
            print("其标签为：",labelResult)
            if labelResult != labelLists[index2]:
                errorNum += 1
        print("错误率为：",errorNum/len(testLists))
    def testBayes2(self):
        '''
        对《机器学习实战》例4进行实验
        与python文件开头的结果进行对比
        :return:
        '''
        trainLists = [[1, 'S'], [1, 'M'], [1, 'M'], [1, 'S'], [1, 'S'], [2, 'S'], [2, 'M'], [2, 'M'],
                     [2, 'L'], [2, 'L'], [3, 'L'], [3, 'M'], [3, 'M'], [3, 'L'], [3, 'L']]
        trainLables = [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]
        vocabLists = self.vocabList(trainLists)
        #print(vocabLists)
        resultList,labelSet = self.BayesTrain(trainLists,trainLables,vocabLists)
        #print(array(resultList))
        label = self.BayesClassify(resultList,labelSet,vocabLists,[2,'S'])
        print("该实例标签是",label)


if __name__ == '__main__':
    #test1 = Bayes1()
    #test1.Bayes_test()
    '''
    1 对应的概率是： 0.032679738562091505
    -1 对应的概率是： 0.061002178649237467
    实例标签是 -1
    '''
    test2 = StaticBayes()
    test2.testBayes2()
    '''
    1 的概率是 -4.548011772148143
   -1 的概率是 -3.8346618842706777
    该实例标签是 -1
    '''
    #test2.testBayes()
