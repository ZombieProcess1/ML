# !/usr/bin/env python
"""A more advanced Reducer, using Python iterators and generators."""

from itertools import groupby
from operator import itemgetter
import sys


def read_mapper_output(file):
    for line in file:
        yield line.rstrip().split()


def main():
    # input comes from STDIN (standard input)
    data = read_mapper_output(sys.stdin)
    #dataResult = [line for line in data] #[['write', '1'], ['the', '1'], ['results', '1'], ['to', '1'], ['to', '1'], ['to', '1'], ['we', '1'], ['we', '1'], ['step,', '1'], ['what', '1'], ['we', '1'], ['output', '1'], ['here', '1'], ['will', '1'], ['be', '1'], ['the', '1'], ['input', '1'], ['for', '1'], ['the', '1'], ['Reduce', '1'], ['step,', '1'], ['i.e.', '1'], ['the', '1'], ['input', '1'], ['for', '1'], ['reducer.py', '1']]
    #print("data result:",dataResult)
    #for a,b in groupby(dataResult,itemgetter(0)):
        #print("a:",a)
        #bResult = [item for item in b]
        #print("b:",bResult)
    # groupby groups multiple word-count pairs by word,
    # and creates an iterator that returns consecutive keys and their group:
    #   current_word - string containing a word (the key)
    #   group - iterator yielding all ["<current_word>", "<count>"] items
    for current_word, group in groupby(data, itemgetter(0)): #得到的是单词，以该单词为键的所有键值对所在的组
        try:
            total_count = sum(int(count) for current_word, count in group)
            print("{}\t{}".format(current_word, total_count))
        except ValueError:
            # count was not a number, so silently discard this item
            pass


if __name__ == "__main__":
    main()