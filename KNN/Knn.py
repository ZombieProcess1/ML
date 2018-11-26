'''
KNN算法：
1、读取文本文件，将其划分为属性样例和对应的标签
2、对于不同属性，需要进行归一化，得到归一化矩阵、最小值元组、极值元组
3、对于单个样例进行预测，需要与最近的k个样例标签进行归类
4、训练样本集合，得到相应的错误率
5、归总1,2,3，写出一个有文字提示的应用方案
'''

from numpy import *
import operator
import random

class Knn():
    def __init__(self):
        pass

    def readFilename(self,filename,n):
        '''
        读取文件，返回属性样例和样例对应的标签列表
        :param filename:
        :param n:
        :return:
        '''
        #文本文件名，属性列数n
        f = open(filename,'r',encoding='utf-8')
        readLines = f.readlines()
        dataSet = zeros((len(readLines),n))
        labels = []
        for line in readLines:
            line = line.strip().split('\t')
            if line[-1] not in labels:
                labels.append(line[-1])
        index = 0
        dataLabels = []
        for line in readLines:
            line = line.strip().split('\t')
            dataSet[index,:] = line[:n]
            dataLabels.append(line[-1])
            index += 1
        return dataSet,dataLabels

    def autoNorm(self,dataSet,n):
        '''
        将属性样例矩阵进行归一化，具体就是每个值减去列对应最小值，除以列的极差
        :param dataSet:
        :param n:
        :return:
        '''
        minValues = dataSet.min(0)
        maxValues = dataSet.max(0)
        ranges = maxValues - minValues
        autoDataSet = tile(minValues,(dataSet.shape[0],1))
        autoDataSet = dataSet - autoDataSet
        autoDataSet = autoDataSet/(tile(ranges,(dataSet.shape[0],1)))
        return autoDataSet,minValues,ranges

    def referenceTest(self,inX,dataSet,dataLabels,k):
        '''
        给定一个属性样例，推理其对应的标签
        :param inX:
        :param dataSet:
        :param dataLabels:
        :param k:
        :return:
        '''
        #inX是没有标签的，需要我们进行推理得到结果
        autoDataSet,minValues,ranges = self.autoNorm(dataSet,dataSet.shape[1])
        autoInX = (inX-minValues)/ranges
        autoDist = ((autoDataSet - tile(autoInX,(dataSet.shape[0],1)))**2).sum(axis=1)**0.5
        labelIndex = autoDist.argsort()  #得到的是排序位置对应的索引,对ndarray进行排序………………………………（注解1）
        labelCount = {}
        for i in range(k):
            label = dataLabels[labelIndex[i]]
            labelCount[label] = labelCount.get(label,0) +1
        labelResult = sorted(labelCount.items(),key = operator.itemgetter(1),reverse= True)  #得到的是一个元组列表
        return labelResult[0][0]

    def referenceErrorRatio(self,filename,n,ratio,k):
        '''
        给定一个文本文件，训练相应的knn模型，以ratio比例的数据测试，看模型准确率
        :param ratio:
        :param k:
        :return:
        '''
        dataSet,dataLabels = self.readFilename(filename,n)
        autoDataSet,minValues,ranges = self.autoNorm(dataSet,n)
        m = int(autoDataSet.shape[0]*ratio)
        #print(autoDataSet.shape[0])
        #随机的生成m个索引值
        indexlist = random.sample(range(0,autoDataSet.shape[0]),m)
        #这里不可以用np.random.randint()因为随机数可能会重复
        #需要注意的是这里的random.sample中的random需要单独import，不然会出现参数超个数的错误……………………(注解2)
        #print(indexlist)
        testDataSet = zeros((m,n))
        testLabels = []
        xunlianDataSet = zeros((autoDataSet.shape[0]-m,n))
        xunlianLabels = []
        for i in range(m):
            testDataSet[i,:] = autoDataSet[indexlist[i],:]
            testLabels.append(dataLabels[indexlist[i]])
        j = 0
        for t in range(autoDataSet.shape[0]):
            if t in indexlist:
                pass
            else:
                #print("j的值是",j,"t的值是",t)
                xunlianDataSet[j,:] = autoDataSet[t,:]
                xunlianLabels.append(dataLabels[t])
                j += 1
        errorRatio = 0
        for inx in range(testDataSet.shape[0]):
            labelX = self.referenceTest(testDataSet[inx],xunlianDataSet,xunlianLabels,k)
            if labelX != testLabels[inx]:
                errorRatio += 1
            print("该样本原本标签是",testLabels[inx],"推理结果为",labelX)
        print("推理错误率为",errorRatio/m)

    def referenceApp(self,filename,n,k):
        arg1 = float(input("请输入第一个参数（飞行常客里程数量）："))
        arg2 = float(input("请输入第二个参数（玩视频游戏所耗时间百分比）："))
        arg3 = float(input("请输入第三个参数（冰淇淋食用量）："))
        inX = array([arg1,arg2,arg3])
        dataSet, dataLabels = self.readFilename(filename, n)
        autoDataSet, minValues, ranges = self.autoNorm(dataSet, n)
        autoInX = (inX-minValues)/ranges
        inXlabel = self.referenceTest(autoInX,autoDataSet,dataLabels,k)
        print("你对该用户的好感很大概率是：",inXlabel)



if __name__ == '__main__':
    test = Knn()
    #test.referenceErrorRatio("data\\datingTestSet.txt",3,0.1,3)
    while True:
        test.referenceApp("data\\datingTestSet.txt",3,3)

'''
后记:需要注意的是注解1和注解2
'''








