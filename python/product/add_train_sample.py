#coding=utf-8
import numpy as np
from numpy import array
from pandas import DataFrame
from collections import Counter
import os, sys, argparse, time, json, random, codecs

# python python/product/add_train_sample.py -i data/product/multi_task_train_label.txt -o data/product/multi_task_add_train_label.txt

def add_train_sample(input_txt, output_txt):
    key_list = []
    fi = open(input_txt, 'r')
    for line in fi:
        category = line.strip().split()[1]
        style    = line.strip().split()[2]
        color    = line.strip().split()[3]
        texture  = line.strip().split()[4]
        key_list.append((category, style, color, texture))
    fi.close()

    key_num_dict = dict(Counter(key_list))
    key_imglist_dict = {key:[] for key in key_num_dict.keys()}
    train_num_mean = int(np.mean(key_num_dict.values()))
    print len(key_num_dict), train_num_mean

    fi = open(input_txt, 'r')
    for line in fi:
        line_list = line.strip().split()
        key_imglist_dict[(line_list[1], line_list[2], line_list[3], line_list[4])].append(line)
    fi.close()
    
    fo = open(output_txt, 'w')
    for key in key_num_dict.keys():
        random.shuffle(key_imglist_dict[key])
        train_lines = key_imglist_dict[key]
        train_num = key_num_dict[key]
        times = train_num_mean/train_num
        if times >= 1:
            train_lines = times*train_lines + random.sample(train_lines, train_num_mean - times*train_num)
        fo.writelines(train_lines)
    fo.close()

def get_args():
    parser = argparse.ArgumentParser(description='analysis json file')
    parser.add_argument('-i', dest='input_txt',
        help='multi_task_train_label.txt', default=None, type=str)  
    parser.add_argument('-o', dest='output_txt',
        help='multi_task_add_train_label.txt', default=None, type=str)   
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

if __name__=='__main__':

    tic = time.time()

    args = get_args()
    input_txt  = args.input_txt
    output_txt = args.output_txt
    add_train_sample(input_txt, output_txt)
    
    toc = time.time()
    print 'running time:{}'.format(toc-tic)