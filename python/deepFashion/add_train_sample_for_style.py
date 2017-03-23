import random
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
from pandas import DataFrame
import os, sys, time, argparse

# python python/deepFashion/add_train_sample_for_style.py -i data/deepFashion/style_train_val_multilabel.txt -n data/deepFashion/others/style_train_val_informations.csv -r models/style_vgg19_bn/results/12w_ft_6w.csv -o data/deepFashion/style_add_train_val_multilabel.txt

def add_train_sample_for_style(input_txt, num_csv, recall_csv, output_txt):
    num_list = array(pd.read_csv(num_csv)['amount'])
    recall_list = array(pd.read_csv(recall_csv)['recall'])
    mean_num = int(np.mean(num_list))
    label_num_dict = {i:mean_num-a for i,a in enumerate(num_list) if a<mean_num and recall_list[i]==0}
    cond_labels = label_num_dict.keys()

    fo = open(output_txt, 'w')

    label_sam_dict = {label:[] for label in cond_labels}
    with open(input_txt, 'r') as fi:
        for line in fi:
            fo.write(line)
            multilabel = map(int, line.strip().split()[1:])
            if sum(multilabel) == 1:
                label = multilabel.index(1)
                if label in cond_labels:
                    label_sam_dict[label].append(line)
    for label in cond_labels:
        if len(label_sam_dict[label])<10:
            with open(input_txt, 'r') as fi:
                for line in fi:
                    multilabel = map(int, line.strip().split()[1:])
                    if multilabel[label]==1 and sum(multilabel)==2:
                        label_sam_dict[label].append(line)
        num = len(label_sam_dict[label])
        times = mean_num//num
        print label, num
        add_lines = label_sam_dict[label]
        add_lines = times*add_lines + random.sample(add_lines, mean_num-times*num)
        fo.writelines(add_lines)
    fo.close()

def get_args():
    parser = argparse.ArgumentParser(description='add train samples for category') 
    parser.add_argument('-i', dest='input_txt',
        help='style_train_val_multilabel.txt', default=None, type=str)
    parser.add_argument('-n', dest='num_csv',
        help='style_train_val_informations.csv', default=None, type=str) 
    parser.add_argument('-r', dest='recall_csv',
        help='recall_results.csv', default=None, type=str)     
    parser.add_argument('-o', dest='output_txt',
        help='style_add_train_val_multilabel.txt', default=None, type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = get_args()
    input_txt  = args.input_txt
    num_csv    = args.num_csv
    recall_csv = args.recall_csv
    output_txt = args.output_txt
    
    tic = time.time()
    add_train_sample_for_style(input_txt, num_csv, recall_csv, output_txt)
    toc = time.time()
    print 'running time:{} seconds'.format(toc-tic)

