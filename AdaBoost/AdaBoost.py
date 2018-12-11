'''
核心思想：三个臭皮匠顶个诸葛亮。
串行训练出多个弱分类器，后面的弱分类器基于上一个弱分类器对每个样本的权重做出调整。最终，为多个弱分类器赋予不同的比重，得到强分类器。
1、初始化每个样本的权重，进行训练，用分类错误率衡量，得到初始的弱分类器。
2、根据错误率，可以求得该弱分类器的比重系数。
3、根据2中系数以及原始样本标签，所得弱分类器预测结果可以重新对每个样本值进行调整
4、重复上面操作进入新一轮的迭代，直到迭代次数达到规定值或者强分类器的分类错误率为0停止
在具体的实验中我会拿《统计学习方法》例8.1的训练数据以及《机器学习实战》上的数据（样本空间一维与二维的展示）
'''
from numpy import *
class AdaBoost():
    def __init__(self):
        pass
    def loadDataSet1(self):
        '''
        《机器学习实战》例
        :return:
        '''
        trainMat = matrix([[1,2.1],[2,1.1],[1.3,1],[1,1],[2,1]])
        classLabels = [1,1,-1,-1,1]
        #print(trainMat)
        return trainMat,classLabels
    def loadDataSet2(self):
        '''《统计学习方法》例'''
        trainMat = matrix([0,1,2,3,4,5,6,7,8,9]).transpose()
        classLabels = [1,1,1,-1,-1,-1,1,1,1,-1]
        #print(trainMat)
        return trainMat,classLabels
    def classify(self,trainMat,dimen,val,flag):
        '''
        该函数用于对某个维度属性按照阈值划分，前后分别对应同类标签，flag决定了具体是哪一类
        :param trainMat: 训练样本集矩阵形式
        :param dimen: 属性维度
        :param val: 划分该属性的阈值
        :param flag: 用于决定阈值前后分类标签是哪一类,LF表示小于阈值都是-1，RF表示大于阈值都是-1标签
        :return: 返回划分的结果
        '''
        retArray = ones((shape(trainMat)[0],1)) #先将划分结果初始化为1
        if flag == 'LF': #小于阈值的部分标签都给-1，否则大于阈值的部分标签都给-1
            retArray[trainMat[:,dimen] <= val] = -1
        else:
            retArray[trainMat[:,dimen] > val] = -1
        return retArray
    def buildStump(self,trainMat,classLabels,D):
        '''
        1、对于不同的属性维进行迭代
        2、不同属性维根据不同的阈值迭代，阈值由该属性值最大最小以及步长决定
        3、根据阈值不同的标签划分方式迭代
        :param trainMat: 训练样本集
        :param classLabels: 标签集
        :param D: 初始的样本权重向量
        :return: 弱分类器（包括划分属性维、阈值、flag决定方式）、该弱分类器错误率，该弱分类器的预测结果
        '''
        dataMatrix = mat(trainMat)
        labelMat = mat(classLabels).T
        m,n = shape(dataMatrix)
        numSteps = 10 #步长，决定了阈值每次的增量
        bestStump = {} #存储弱分类器各属性值
        bestClasEst = mat(zeros((m,1))) #存储弱分类器划分结果
        minError = inf
        for i in range(n): #属性纬度
            valMin = dataMatrix[:,i].min()
            valMax = dataMatrix[:,i].max()
            stepSize = (valMax-valMin)/numSteps #阈值增量
            for j in range(-1,numSteps+1):
                val = valMin + stepSize*j
                for flag in ('LF','RF'):
                    retArray = self.classify(dataMatrix,i,val,flag)
                    errArr = mat(ones((m,1)))
                    errArr[retArray==labelMat] = 0  #将分类结果中错误的置为1，结合权重向量计算下面的错误率
                    weightedError = D.T * errArr
                    #print("属性维度：",i," 划分阈值是：",val," 所用的标签分类方法是：",flag,"该分类的错误率是%.3f："%(weightedError))
                    if weightedError < minError:
                        minError = weightedError
                        bestClasEst = retArray.copy() #最好的划分标签结果
                        bestStump['dimen'] = i #最好的属性维度
                        bestStump['val'] = val #最好的划分阈值
                        bestStump['flag'] = flag #最好的标签分配方式
        return bestStump,bestClasEst,minError
    def adaBoostTrainDS(self,dataMat,classLabels,numIter = 50):
        '''
        完整的训练直到得到强分类器的过程：
        1、每一轮得到一个弱分类器以及相应的标签结果、分类错误率
        2、根据错误率计算该弱分类器的权重系数 -----将该弱分类器相关的属性存入列表中
        3、根据权重系数、原标签、分类标签调整每个样本的权重
        4、达到迭代次数，或者强分类器的分类错误率为0，则结束迭代计算 -----强分类器的分类错误率前后叠加
        :param dataMat: 训练样本集
        :param classLabels: 原始标签
        :param numIter: 迭代次数，预设默认值50
        :return: 最终得到的强分类器
        '''
        dataMatrix = mat(dataMat)
        labelMatrix = mat(classLabels).T
        weakClassify = []  #用于记录所有的弱分类器
        m = shape(dataMatrix)[0]
        strongError = mat(zeros((m,1))) #记录每轮迭代后强分类器的错误率，借此求出平均错误率，当为0时，结束迭代
        D = mat(ones((m,1))/m)  #初始化每个元素的权重
        for iter in range(numIter):
            bestStump,bestClasEst,minError = self.buildStump(dataMat,classLabels,D)  #一次训练得到的弱分类器
            alpha = float(0.5*log((1-minError)/max(minError,1e-16))) #该弱分类器的系数
            #print("D:",D.T)
            bestStump['alpha'] = alpha #往该弱分类器的属性信息中添加系数值
            #print("alpha:",alpha)
            weakClassify.append(bestStump)  #存储该弱分类器，方便以后进行调用
            weightVal = multiply(-1 *alpha* labelMatrix, bestClasEst ) #开始调整样例权重的分子值
            D = multiply(D, exp(weightVal))
            D = D / D.sum()
            strongError +=alpha*bestClasEst  #强分类器加上每一轮迭代的权重乘以对应分类标签值
            #print("strongError:",strongError.T)
            errorRate = multiply(sign(strongError)!=labelMatrix,ones((m,1))).sum()/m
            #print("errorRate:",errorRate)
            #print()
            if errorRate == 0.0:
                break
        return  weakClassify
    def testClassify(self,testDataMat,weakClassify):
        '''
        对于输入的测试样本集合，根据训练出的强分类器输出每个样本对应的分类标签
        具体做法，针对每个弱分类器，都可以根据该分类器的阈值、标签划分方法、属性维度得出测试集对应的标签，然后用标签乘以权重就是该弱分类器的测试结果，多个分类器加权结果就是总的预测结果
        :param testDataMat: 测试样本集
        :param weakClassify: 弱分类器加权重组合的强分类器字典
        :return: 每个测试样本的分类标签
        '''
        testDataMatrix = mat(testDataMat)
        resultArr = mat(zeros((shape(testDataMatrix)[0],1)))
        for i in range(len(weakClassify)):
            testArr = self.classify(testDataMatrix,weakClassify[i]['dimen'],weakClassify[i]['val'],weakClassify[i]['flag'])
            resultArr += testArr*weakClassify[i]['alpha']
            print("叠加弱分类器结果:",resultArr)
        return sign(resultArr)

if __name__ == '__main__':
    test = AdaBoost()
    #trainMat ,classLabels = test.loadDataSet2()
    trainMat, classLabels = test.loadDataSet1()
    weakClassify = test.adaBoostTrainDS(trainMat,classLabels,9)
    resultLabel = test.testClassify([0,0],weakClassify)
    print(resultLabel)
