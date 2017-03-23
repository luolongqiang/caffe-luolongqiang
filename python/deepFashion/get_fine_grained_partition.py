#!/usr/bin/python
# -*- coding: <utf-8> -*-
import random
import numpy as np
from numpy import array
import os, sys, time, argparse
from collections import Counter
import matplotlib.pyplot as plt
from pandas import DataFrame

# python python/deepFashion/get_fine_grained_partition.py -i data/deepFashion/Anno/list_category_img.txt -o data/deepFashion -c data/deepFashion/fined_grained_count_result.csv

def get_dir_dict(input_txt):     
    dir_list = []
    with open(input_txt, 'r') as fi:
        for line in list(fi)[2:]: 
            file_name = line.split()[0]
            dir_list.append(file_name[:file_name.rfind('/')])
    fi.close()
    class_num_dict = dict(Counter(dir_list))
    for i, cls in enumerate(class_num_dict.keys()):
        class_num_dict[cls] = (i, class_num_dict[cls])
    return class_num_dict

def output_results(class_num_dict, output_csv):
    class_set = array(class_num_dict.keys())
    label_set = array(class_num_dict.values())[:,0]
    sample_num = array(class_num_dict.values())[:,1]
    df = DataFrame({'class':class_set, 'label':label_set, 'sample_num':sample_num})
    df = df[['class', 'label', 'sample_num']]
    df.to_csv(output_csv, index = False)

    plt.plot(label_set, sample_num, color = "blue")
    plt.xlabel("class")
    plt.ylabel("sample amount")
    plt.title("sample amount of class")
    plt.xlim(0, label_set[-1] + 1)
    plt.ylim(0, max(sample_num) + 5)
    plt.savefig(output_csv.replace('csv','jpg'), dpi=300)
    plt.close()

def get_partition_datasets(class_num_dict, input_txt, output_dir):
    class_set = array(class_num_dict.keys())
    sample_num = array(class_num_dict.values())[:,1]
    train_num = array(sample_num*0.8, dtype = np.int)
    val_num = array(sample_num*0.1, dtype = np.int)
    test_num = sample_num - train_num - val_num
    partition_num = zip(train_num, val_num, test_num)
    class_partition_num = dict(zip(class_set, partition_num))

    class_partition_dict = {cls:[] for cls in class_set}
    with open(input_txt, 'r') as fi:
        for line in list(fi)[2:]:
            file_name = line.split()[0]
            new_file_name = 'data/deepFashion/crop_img' + file_name[3:]
            cls = file_name[:file_name.rfind('/')]
            label = str(class_num_dict[cls][0])
            class_partition_dict[cls].append(new_file_name + ' ' + label + '\n')
    fi.close()

    fo_train = open(output_dir+'/fine_grained_train_label.txt', 'w')
    fo_val   = open(output_dir+'/fine_grained_val_label.txt', 'w')
    fo_test  = open(output_dir+'/fine_grained_test_label.txt', 'w')
    for cls in class_set:
        random.shuffle(class_partition_dict[cls])
        train_num, val_num, test_num = class_partition_num[cls]
        train_lines = array(class_partition_dict[cls])[:train_num]
        val_lines   = array(class_partition_dict[cls])[train_num:train_num+val_num]
        test_lines  = array(class_partition_dict[cls])[train_num+val_num:]
        fo_train.writelines(train_lines)
        fo_val.writelines(val_lines)
        fo_test.writelines(test_lines) 
    fo_train.close()
    fo_val.close()
    fo_test.close()
    #end

def get_args():
    parser = argparse.ArgumentParser(description='get style sample informations') 
    parser.add_argument('-i', dest='input_txt',
        help='list_category_img.txt', default=None, type=str) 
    parser.add_argument('-o', dest='output_dir',
        help='output_partition_dir', default=None, type=str)
    parser.add_argument('-c', dest='output_csv',
        help='category_results.csv', default=None, type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = get_args()
    input_txt = args.input_txt
    output_dir = args.output_dir

    class_num_dict = get_dir_dict(input_txt)
    get_partition_datasets(class_num_dict, input_txt, output_dir)

    if args.output_csv:
        output_results(class_num_dict, args.output_csv)