import numpy as np
from numpy import array
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
from pandas import DataFrame
import os, sys, time, argparse

# python python/deepFashion/get_attr10_partition.py -d data/deepFashion/others/attr_distr_results.csv -m data/deepFashion/attr_train_multilabel.txt -n data/deepFashion/attr10_train_multilabel.txt
# scp /home/luolongqiang/deepdraw/caffe/python/deepFashion/get_attr10_partition.py luolongqiang@192.168.1.182:/home/luolongqiang/deepdraw/caffe/python/deepFashion/get_attr10_partition.py

def get_attr10_data(attr_distr_csv, multilabel_txt, new_multilabel_txt):     
    attr_distr = pd.read_csv(attr_distr_csv)
    cond_index = array(attr_distr['num']>=10000)
    fi = open(multilabel_txt, 'r')
    fo = open(new_multilabel_txt, 'w')
    for i,line in enumerate(fi):
        print i
        img_file_name = line.split()[0]
        multilabel = array(map(int, line.strip().split()[1:]))
        cond_multilabel = multilabel[cond_index]
        if sum(cond_multilabel)>0:
            new_line = img_file_name
            for label in cond_multilabel:
                new_line += ' ' + str(label)
            fo.write(new_line+'\n')
    fi.close()
    fo.close()

def get_args():
    parser = argparse.ArgumentParser(description='count attribution informations') 
    parser.add_argument('-d', dest='distr_csv',
        help='attr_distr_results.csv', default=None, type=str)
    parser.add_argument('-m', dest='multilabel_txt',
        help='attr_multilabel.txt', default=None, type=str)  
    parser.add_argument('-n', dest='new_multilabel_txt',
        help='attr10_multilabel.txt', default=None, type=str)  
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = get_args()
    attr_distr_csv     = args.distr_csv
    multilabel_txt     = args.multilabel_txt
    new_multilabel_txt = args.new_multilabel_txt
    
    tic = time.clock()
    get_attr10_data(attr_distr_csv, multilabel_txt, new_multilabel_txt)
    toc = time.clock()
    print 'running time:{} seconds'.format(toc-tic)
