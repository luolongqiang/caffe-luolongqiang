#coding=utf-8
import numpy as np
from numpy import array
from pandas import DataFrame
from collections import Counter
import os, sys, argparse, time, json, random, codecs

# python python/product/count_label.py -t data/product/product_list.txt -c data/product/category_count.csv -k category

def count_and_partition(txt_file, csv_file, key_word, parti_dir):
    word_dict = {'category':1, 'style':2, 'color':3, 'texture':4}
    word_index = word_dict[key_word]
    key_list = []
    with open(txt_file, 'r') as fi:
        for line in fi:
            clas = line.strip().split()[word_index]
            key_list.append(int(clas))

    key_num_dict = dict(Counter(key_list))
    key_set = np.sort(np.array(key_num_dict.keys(), dtype=np.int))
    num_set = [key_num_dict[clas] for clas in key_set]
    if csv_file:
        df = DataFrame({key_word:key_set, 'num':num_set})
        df = df[[key_word, 'num']]
        df.to_csv(csv_file, index=False, encoding='utf-8')
    '''
    key_imglist_dict = {clas:[] for clas in key_set}
    key_label_dict = {clas:str(i) for i, clas in enumerate(key_set)}
    with open(txt_file, 'r') as fi:
        for line in fi:
            line_list = line.strip().split()
            clas = line_list[word_index]
            new_line = line_list[0] + ' ' + key_label_dict[clas] + '\n'
            key_imglist_dict[clas].append(new_line)
    
    fo_train = open(parti_dir + '/' + key_word + '_train_label.txt', 'w')
    fo_val   = open(parti_dir + '/' + key_word + '_val_label.txt',   'w')
    fo_test  = open(parti_dir + '/' + key_word + '_test_label.txt',  'w')
    for clas in key_set:
        random.shuffle(key_imglist_dict[clas])
        total_num = key_num_dict[clas]
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
        train_lines = array(key_imglist_dict[clas])[:train_num]
        val_lines   = array(key_imglist_dict[clas])[train_num:train_num+val_num]
        test_lines  = array(key_imglist_dict[clas])[train_num+val_num:]
        fo_train.writelines(train_lines)
        fo_val.writelines(val_lines)
        fo_test.writelines(test_lines) 
    fo_train.close()
    fo_val.close()
    fo_test.close()
    '''

def get_args():
    parser = argparse.ArgumentParser(description='analysis json file')
    parser.add_argument('-t', dest='txt_file',
        help='product_list.txt', default=None, type=str)
    parser.add_argument('-c', dest='csv_file',
        help='count.csv', default=None, type=str) 
    parser.add_argument('-k', dest='key_word',
        help='key_word: e.g, category\style\color\texture', default=None, type=str)   
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
    csv_file  = args.csv_file
    key_word  = args.key_word
    parti_dir = args.parti_dir
    
    count_and_partition(txt_file, csv_file, key_word, parti_dir)
    
    toc = time.time()
    print 'running time:{}'.format(toc-tic)