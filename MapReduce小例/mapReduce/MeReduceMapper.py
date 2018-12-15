import sys
from numpy import *
def read_input(file):
    for line in file:
        yield line.rstrip()


input = read_input(sys.stdin)
input = [float(line) for line in input]
numInputs = len(input)
input = mat(input)
sqInput = power(input, 2)
print("{}\t{}\t{}".format(numInputs, mean(input) ,mean(sqInput)))
sys.stderr.write("report:still alive")