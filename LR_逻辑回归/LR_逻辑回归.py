'''
所谓逻辑回归实际上就是一个分类问题，针对输入的数值型数据选择最佳的参数将分类边界拟合为一条直线。
用属性值、参数值计算的结果输入sigmoid函数，得到的结果大于0.5，那么归为一类，否则归为另一类
在具体求解参数的过程中，用到了梯度上升算法，所谓梯度就是对变量的偏导。放在具体分类问题中，为了找到最好的直线，
其实就是一个样例分类出了差错，那么就把该直线往正确的方向挪一点点，直到所有的点最优的分布在直线两端或达到迭代次数时结束。
具体步骤
1、读取文件中数据集，处理为列表形式
2、给出sigmoid函数的求解函数
3、针对所有训练集迭代归类，调参
4、对梯度上升算法进行改进，用到模拟退火的思想，动态变化参数alpha；随机选取样本调整——避免局部最优、避免周期波动
5、对马患病生死预测的应用
'''
from numpy import *
import random
import matplotlib.pyplot as plt

class LR():
    def __init__(self):
        pass
    def loadDataSet(self,columns,filename):
        '''
        :param columns: 属性列数（输入维度）
        :param filename: 文不能文件名
        :return: 样例及对应标签分类的列表集合
        '''
        lines = open(filename,'r',encoding='utf-8').readlines()
        dataLists = []
        labelLists = []
        for line in lines:
            line = line.strip().split('\t')
            data = [1]
            for i in range(columns):
                data.append(float(line[i]))
            dataLists.append(data)
            labelLists.append(float(line[columns]))
        return dataLists,labelLists  #返回的都是列表形式
    def sigmoid(self,inX):
        '''
        输入属性列表，返回对应的sigmoid函数值
        :param inX: 该值属性 float
        :return: sigmoid值  float
        '''
        return 1/(1+exp(-inX))
    def train_Lr(self,trainLists,labelLists,iter,a):
        '''
        输入均为列表形式，请自行转换为matrix或者array,但是要注意区分应用，前者直接是矩阵运算，后者元组一一对应计算需要sum()
        :param trainLists: 训练集
        :param labelLists: 训练标签
        :return: weights参数列表  list
        '''
        trainMat = mat(trainLists)  #区分array
        labelMat = mat(labelLists).transpose()
        m,n = shape(trainMat)
        weights = ones((n,1)) #列向量
        for i in range(iter):
            inX = trainMat*weights
            result = self.sigmoid(inX)
            error = labelMat-result  #是不断往正确结果靠近，所以需要注意是标准结果减去运行后结果，否则最后的weights训练结果会不一样
            weights = weights + a*trainMat.transpose()*error  #需要转置，否则行列不对应
        return weights
    def train_lr_ran(self,trainLists,labelLists,a):
        '''
        用一个一个的样例去模拟计算
        :param trainLists:
        :param labelLists:
        :param a:alpha
        :return:
        '''
        trainMat = array(trainLists)  #array与mat的区别还是很大的，不要混淆使用
        m,n = shape(trainMat)
        weights = ones(n) #行向量
        for i in range(m):
            inX = sum(trainMat[i]*weights)
            result = self.sigmoid(inX)
            error = labelLists[i] - result
            weights += a*error*trainMat[i]
        return weights
    def train_lr_ran_iter(self,trainLists,labelLists,iter):
        '''
        1、多次迭代，每次迭代均改变alpha值
        2、一轮迭代随机选择样例进行计算
        :param trainLists:
        :param labelLists:
        :param iter: 自定义的迭代次数
        :return: weights
        '''
        trainMat = array(trainLists)
        m,n = shape(trainMat)
        weights = ones(n)
        for i in range(iter):
            dataIndex = list(arange(m))  #list()转为列表形式，是因为下面的del()只适用于列表形式
            for j in range(m):
                a = 4/(1+i+j) + 0.01
                randIndex = int(random.uniform(0,len(dataIndex)))  #该处用uniform生成的是浮点数，所以需要转换为int
                inX = sum(trainMat[randIndex] * weights)
                result = self.sigmoid(inX)
                error = labelLists[randIndex] - result
                weights = weights+a*error*trainMat[randIndex]
                del(dataIndex[randIndex])
        return weights
    def lr_show(self,trainLists,labelLists,weights):
        '''
        根据给定的数据集以及运行得到的参数绘制分类图像
        '''
        fig = plt.figure()
        ax = fig.add_subplot(111)
        x0 = []
        y0 = []
        x1 = []
        y1 = []
        for i in range(len(trainLists)):
            if labelLists[i] == 1:
                x1.append(trainLists[i][1])  #这里不要把[i]给弄掉了
                y1.append(trainLists[i][2])
            if labelLists[i] == 0:
                x0.append(trainLists[i][1])
                y0.append(trainLists[i][2])
        ax.scatter(x0,y0,s=30,c='red',marker='o')
        ax.scatter(x1,y1,s=30,c='green',marker='o')
        if type(weights).__name__ == 'matrix':
            weights = weights.getA()  #array,但每个元素依然是array类型，所以需要将其一一读出存入list，然后再转回array类型
            weights = array([weights[i][0] for i in range(len(weights))])
        x = arange(-3,3,0.1)
        y = -(weights[0] + weights[1]*x)/weights[2]
        plt.plot(x,y)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()
    def test(self):
        filename = 'data\\testSet.txt'
        trainLists ,labelLists = self.loadDataSet(2,filename)
        weights = self.train_Lr(trainLists,labelLists,500,0.001)
        weights1 = self.train_lr_ran(trainLists,labelLists,0.01)
        weights2 = self.train_lr_ran_iter(trainLists,labelLists,150)
        #print(type(weights))
        self.lr_show(trainLists,labelLists,weights)
        self.lr_show(trainLists,labelLists,weights1)
        self.lr_show(trainLists,labelLists,weights2)
        #print(weights1)
        #print(weights2)
    def horse_train_lr(self):
        '''
        对生病的马的生死预测
        '''
        trainLists,trainLabels = self.loadDataSet(21,'data\\horse\\horseColicTest.txt')
        testLists,testLabels = self.loadDataSet(21,'data\\horse\\horseColicTraining.txt')
        weights = self.train_lr_ran_iter(trainLists,trainLabels,500)  #array
        testMat = array(testLists)
        errorNum = 0
        for i in range(len(testLists)):
            result = self.sigmoid(sum(testMat[i] * weights))
            if result>0.5:
                classR = 1
            else:classR = 0
            print("该样本原始分类为",testLabels[i],"求取结果分类为",classR)
            if classR != testLabels[i]:
                errorNum += 1
        print("对于马生死预测的错误率为：",errorNum/len(testLists))

if __name__ == '__main__':
    test = LR()
    test.test()
    test.horse_train_lr()
