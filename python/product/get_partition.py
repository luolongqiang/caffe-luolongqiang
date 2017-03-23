#coding=utf-8
import numpy as np
from numpy import array
from pandas import DataFrame
from collections import Counter
import os, sys, argparse, time, json, random, codecs

# python python/product/get_partition.py -t data/product/product_list.txt -p data/product

def count_and_partition(txt_file, parti_dir):
    category_list = []
    style_list = []
    color_list = []
    texture_list = []
    with open(txt_file, 'r') as fi:
        for line in fi:
            category = line.strip().split()[1]
            style    = line.strip().split()[2]
            color    = line.strip().split()[3]
            texture  = line.strip().split()[4]
            category_list.append(category)
            style_list.append(style)
            color_list.append(color)
            texture_list.append(texture)

    category_num_dict = dict(Counter(category_list))
    category_set = np.sort(array(category_num_dict.keys(), dtype=np.int))
    style_set = np.sort(array(list(set(style_list)), dtype=np.int))
    color_set = np.sort(array(list(set(color_list)), dtype=np.int))
    texture_set = np.sort(array(list(set(texture_list)), dtype=np.int))
    print len(category_set), len(style_set), len(color_set), len(texture_set)

    category_imglist_dict = {str(category):[] for category in category_set}
    category_label_dict = {str(category):str(i) for i, category in enumerate(category_set)}
    style_label_dict = {str(style):str(i) for i, style in enumerate(style_set)}
    color_label_dict = {str(color):str(i) for i, color in enumerate(color_set)}
    texture_label_dict = {str(texture):str(i) for i, texture in enumerate(texture_set)}

    with open(txt_file, 'r') as fi:
        for line in fi:
            line_list = line.strip().split()
            a = category_label_dict[line_list[1]]
            b = style_label_dict[line_list[2]]
            c = color_label_dict[line_list[3]]
            d = texture_label_dict[line_list[4]]
            new_line = line_list[0] + ' ' + a + ' ' + b + ' ' + c + ' ' + d + '\n'
            category_imglist_dict[line_list[1]].append(new_line)
    
    fo_train = open(parti_dir + '/multi_task_train_label.txt', 'w')
    fo_val   = open(parti_dir + '/multi_task_val_label.txt',   'w')
    fo_test  = open(parti_dir + '/multi_task_test_label.txt',  'w')
    for category in category_set:
        category = str(category)
        random.shuffle(category_imglist_dict[category])
        total_num = category_num_dict[category]
        if total_num == 1:
            train_num, val_num, test_num == 1, 0, 0
        elif total_num == 3:
            train_num, val_num, test_num == 1, 1, 1
        elif total_num == 4:
            train_num, val_num, test_num == 2, 1, 1
        elif total_num == 5:
            train_num, val_num, test_num == 3, 1, 1
        else:
            train_num = int(total_num*0.8)
            val_num = int(round(total_num*0.1))
            test_num = total_num - train_num - val_num
        train_lines = array(category_imglist_dict[category])[:train_num]
        val_lines   = array(category_imglist_dict[category])[train_num:train_num+val_num]
        test_lines  = array(category_imglist_dict[category])[train_num+val_num:]
        fo_train.writelines(train_lines)
        fo_val.writelines(val_lines)
        fo_test.writelines(test_lines) 
    fo_train.close()
    fo_val.close()
    fo_test.close()

def get_args():
    parser = argparse.ArgumentParser(description='analysis json file')
    parser.add_argument('-t', dest='txt_file',
        help='product_list.txt', default=None, type=str)  
    parser.add_argument('-p', dest='parti_dir',
        help='partition_dir', default=None, type=str)   
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

if __name__=='__main__':

    tic = time.time()

    args = get_args()
    txt_file  = args.txt_file
    parti_dir = args.parti_dir
    count_and_partition(txt_file, parti_dir)
    
    toc = time.time()
    print 'running time:{}'.format(toc-tic)