import random
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
from collections import Counter
from pandas import DataFrame
import os, sys, time, argparse

#python python/deepFashion/add_sample_for_category.py -i data/deepFashion/category_train_label.txt -o data/deepFashion/category_add_train_label.txt

def get_addtrain_samples(input_txt, output_txt):
    label_list = []
    with open(input_txt, 'r') as fi:
        for line in fi: 
            label_list.append(line.strip().split()[-1])
    fi.close()
    label_num_dict = dict(Counter(label_list))
 
    labels = set(label_list)
    label_partition_dict = {label:[] for label in labels}
    with open(input_txt, 'r') as fi:
        for line in fi:
            label = line.strip().split()[-1]
            label_partition_dict[label].append(line)
    fi.close()

    train_num_mean = int(np.mean(label_num_dict.values()))
    fo = open(output_txt, 'w')
    for label in labels:
        random.shuffle(label_partition_dict[label])
        train_lines = label_partition_dict[label]
        train_num = label_num_dict[label]
        times = train_num_mean/train_num
        if times >= 1:
            train_lines = times*train_lines + random.sample(train_lines, train_num_mean - times*train_num)
        fo.writelines(train_lines)
    fo.close()         

def get_args():
    parser = argparse.ArgumentParser(description='add train samples for category') 
    parser.add_argument('-i', dest='input_txt',
        help='category_train_label.txt', default=None, type=str) 
    parser.add_argument('-o', dest='output_txt',
        help='category_add_train_label.txt', default=None, type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = get_args()
    input_txt = args.input_txt
    output_txt = args.output_txt
    
    tic = time.clock()
    get_addtrain_samples(input_txt, output_txt)
    toc = time.clock()
    print 'running time:{} seconds'.format(toc-tic)

