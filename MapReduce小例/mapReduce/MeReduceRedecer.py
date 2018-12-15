import sys
from numpy import *
def read_input(file):
    for line in file:
        yield line.rstrip()  #yield类似于return 只不过返回的是迭代器类型，也就是可以用yield.next()来访问后面的数据

input = read_input(sys.stdin)
mapperOut = [line.split('\t') for line in input]
print("mapperOut:",mapperOut)
cumVal = 0.0
cumSumSq = 0.0
cumN = 0.0
for instance in mapperOut:
    print("instance",instance)
    print("instance[0]",instance[0])
    print("instance[1]",instance[1])
    print("instance[2]",instance[2])
    nj = float(instance[0])
    cumN += nj
    cumVal += nj*float(instance[1])
    cumSumSq += nj*float(instance[2])
    mean = cumVal/cumN
    varSum = (cumSumSq - 2*mean*cumVal + cumN*mean*mean)/cumN
    print(cumN,cumVal,cumSumSq)
    sys.stderr.write("report:still alive")
