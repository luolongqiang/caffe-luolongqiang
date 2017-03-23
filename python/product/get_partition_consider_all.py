#coding=utf-8
import numpy as np
from numpy import array
from pandas import DataFrame
from collections import Counter
import os, sys, argparse, time, json, random, codecs

# python python/product/get_partition_consider_all.py -t data/product/product_list.txt -p data/product

def count_and_partition(txt_file, parti_dir):
    key_list = []
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
            key_list.append((category, style, color, texture))
            category_list.append(category)
            style_list.append(style)
            color_list.append(color)
            texture_list.append(texture)

    key_num_dict = dict(Counter(key_list))
    category_set = np.sort(list(set(category_list)))
    style_set = np.sort(list(set(style_list)))
    color_set = np.sort(list(set(color_list)))
    texture_set = np.sort(list(set(texture_list)))
    print len(key_num_dict)

    key_imglist_dict = {key:[] for key in key_num_dict.keys()}
    category_label_dict = {category:str(i) for i, category in enumerate(category_set)}
    style_label_dict = {style:str(i) for i, style in enumerate(style_set)}
    color_label_dict = {color:str(i) for i, color in enumerate(color_set)}
    texture_label_dict = {texture:str(i) for i, texture in enumerate(texture_set)}

    with open(txt_file, 'r') as fi:
        for line in fi:
            line_list = line.strip().split()
            a = category_label_dict[line_list[1]]
            b = style_label_dict[line_list[2]]
            c = color_label_dict[line_list[3]]
            d = texture_label_dict[line_list[4]]
            new_line = line_list[0] + ' ' + a + ' ' + b + ' ' + c + ' ' + d + '\n'
            key_imglist_dict[(line_list[1], line_list[2], line_list[3], line_list[4])].append(new_line)
    
    fo_train = open(parti_dir + '/multi_task_train_label.txt', 'w')
    fo_val   = open(parti_dir + '/multi_task_val_label.txt',   'w')
    fo_test  = open(parti_dir + '/multi_task_test_label.txt',  'w')
    for key in key_num_dict.keys():
        random.shuffle(key_imglist_dict[key])
        total_num = key_num_dict[key]
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
        train_lines = array(key_imglist_dict[key])[:train_num]
        val_lines   = array(key_imglist_dict[key])[train_num:train_num+val_num]
        test_lines  = array(key_imglist_dict[key])[train_num+val_num:]
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