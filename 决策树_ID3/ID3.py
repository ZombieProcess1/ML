'''
1、决策树的构建过程需要知道如何求解熵
2、根据属性列求取对应的信息增益
3、以最大的信息增益对应的属性列划分数据集，得到数据子集，该数据子集去掉了对应的属性列
4、递归第三步构造决策树，边界条件是属性用完或者分类单一
5、存储得到的决策树，以及加载决策树，避免重复训练
6、给定属性列集合进行预测
'''
from  math import  log
class ID3():
    def __init__(self):
        pass

    def getDataSet(self,filename):
        '''
        :param filename: 文本文件名
        :return: 返回样例列表
        '''
        f = open(filename)
        readlines = f.readlines()
        dataSet = []
        for line in readlines:
            line = line.strip().split('\t')
            #print(type(line))  #list
            dataSet.append(line)
        return dataSet

    def InfoShannon(self,dataSet):
        '''
        :param dataSet: 样例列表集合
        :return: 返回熵
        '''
        dataNum = len(dataSet)
        labelCount = {}
        for data in dataSet:
            if data[-1] not in labelCount.keys():
                labelCount[data[-1]] = 0
            labelCount[data[-1]]  += 1
        shannon = 0
        for key in labelCount.keys():
            pro = labelCount[key]/dataNum
            shannon -= pro*log(pro,2)
        return shannon

    def splitData(self,dataSet,axis,value):
        '''
        :param dataSet: 给定数据集
        :param axis: 属性列索引
        :param value: 属性列的值
        :return: 返回以该属性列属性值划分得到的部分数据子集，子集剔除了该属性列
        '''
        subDataSet = []
        for data in dataSet:
            if data[axis] == value:
                subData = data[:axis]
                subData.extend(data[axis+1:])
                subDataSet.append(subData)
        return subDataSet

    def bestFeatureToSplitDataSet(self,dataSet):
        '''
        :param dataSet:数据集
        :return: 最好的划分数据的属性列索引
        '''
        labelNum = len(dataSet[0])-1
        dataSetNum = len(dataSet)
        baseShannon = self.InfoShannon(dataSet)
        bestFeature = -1
        bestInfoGain = 0
        for i in range(labelNum):
            labelValues = [example[i] for example in dataSet]
            uniqueValues = set(labelValues)
            newShannon = 0
            for value in uniqueValues:
                subDataSet = self.splitData(dataSet,i,value)
                #print(subDataSet)
                pro = len(subDataSet)/dataSetNum
                newShannon += pro*self.InfoShannon(subDataSet)
            if baseShannon-newShannon>bestInfoGain:
                bestInfoGain = baseShannon-newShannon
                bestFeature = i
        return bestFeature

    def selectLabel(self,dataSet):
        '''
        选举投票类别
        :param dataSet: 给定数据集
        :return: 类别名称
        '''
        labelCount = {}
        for data in dataSet:
            if data[-1] not in labelCount.keys():
                labelCount[data[-1]] = 0
            labelCount[data[-1]] += 1
        labels = sorted(labelCount.items(),key= lambda item:item[1],reverse=True)
        return labels[0][0]

    def createTree(self,dataSet,attributes):
        '''
        1、按照最优属性分类子集
        2、子集属于同一类或属性用完，则返回类别
        3、其中属性用完，子集不是一类，则用选举投票方式
        4、递归子集
        :param dataSet:给定数据集
        :param attributes: 属性列名，需要与数据集列一一对应
        :return: 返回构造的字典形式的决策树
        '''
        classList = [example[-1] for example in dataSet]
        if classList.count(classList[0]) == len(dataSet):
            return classList[0]
        if len(dataSet[0]) == 1:
            return self.selectLabel(dataSet)
        bestFeature = self.bestFeatureToSplitDataSet(dataSet) #仅仅获得数据集对应的属性列
        attributeName = attributes[bestFeature]
        myTree = {attributeName:{}}
        del (attributes[bestFeature])
        values = [example[bestFeature] for example in dataSet]
        uniqueValues = set(values)
        for value in uniqueValues:
            subAttributes = attributes[:]
            subDataSet = self.splitData(dataSet,bestFeature,value)
            myTree[attributeName][value] = self.createTree(subDataSet,subAttributes)
        return myTree

    def storeTree(self,myTree,filename):
        '''
        硬盘存储树
        :param myTree:字典结构的树
        :param filename: 文件名
        :return: 无
        '''
        import pickle
        fw = open(filename,'wb')    #注解：该处写入文件时，会出现TypeError: write() argument must be str, not bytes的错误，所以用二进制写入，二进制读出
        pickle.dump(myTree,fw)
        fw.close()

    def loadTree(self,filename):
        import pickle
        fr = open(filename,'rb')
        myTree = pickle.load(fr)
        fr.close()
        return myTree

    def classify(self,myTree,attributes,testVec):
        '''
        给定数据集训练得到决策树，预测给定的向量的分类
        :param myTree: 决策树
        :param attributes:与测试向量属性列一一对应的属性标签
        :param testVec: 测试向量
        :return: 测试向量的分类
        '''
        attribute0 = list(myTree.keys())[0] #得到决策树的根属性名
        attributeIndex = attributes.index(attribute0)
        subKeys = myTree[attribute0].keys()
        for key in subKeys:
            if testVec[attributeIndex] == key:
                subTree = myTree[attribute0][key]
                if type(subTree).__name__=='dict':
                    classLabel = self.classify(subTree,attributes,testVec)
                else:
                    classLabel = subTree
                return  classLabel


if __name__ == '__main__':
    test = ID3()

    #dataSet = test.getDataSet("data\\id3\\lenses.txt")
    attributes = ['age','prescript','astigmatic','tearRate']
    #myTree = test.createTree(dataSet,attributes)   #只用执行一次，下面存入磁盘同理
    #print(type(myTree))   dict
    #test.storeTree(myTree,"data\\id3\\testTree.txt")

    myTree = test.loadTree("data\\id3\\testTree.txt")
    #print(myTree)
    #print(type(myTree))
    #classLabel = test.classify(myTree,attributes,['young','	myope','no','reduced'])
    classLabel = test.classify(myTree,attributes,['pre','myope','yes','normal'])
    print(classLabel)


