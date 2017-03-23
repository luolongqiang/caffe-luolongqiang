import random
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
from collections import Counter
from pandas import DataFrame
import os, sys, time, argparse

def get_category_label_num_dict(input_txt):     
    label_list = []
    with open(input_txt, 'r') as fi:
        for line in list(fi)[2:]: 
            label_list.append(line.strip().split()[-1])
    fi.close()
    label_num_dict = dict(Counter(label_list))
    return label_num_dict

def get_label_partition_num(label_num_dict):
    indx = array(label_num_dict.values())>10000
    cond_label = array(label_num_dict.keys())[indx]
    cond_sample_num = array(label_num_dict.values())[indx]
    #cond_label_to_new = dict(zip(cond_label, range(len(cond_label))))
    train_num = array(cond_sample_num*0.8, dtype = np.int)
    val_num = array(cond_sample_num*0.1, dtype = np.int)
    test_num = cond_sample_num - train_num - val_num
    num_partition = zip(train_num, val_num, test_num)
    label_partition_num = dict(zip(cond_label, num_partition))
    return label_partition_num

def get_partition_datasets(label_num_dict, input_txt, output_dir):
    label_partition_num = get_label_partition_num(label_num_dict)
    labels = label_partition_num.keys()
    label_partition_dict = {label:[] for label in labels}
    with open(input_txt, 'r') as fi:
        for line in list(fi)[2:]:
            label = line.strip().split()[-1]
            if label in labels:
               label_partition_dict[label].append('data/deepFashion/'+line)
    fi.close()

    fo_train = open(output_dir+'/category_train_label.txt', 'w')
    fo_val   = open(output_dir+'/category_val_label.txt', 'w')
    fo_test  = open(output_dir+'/category_test_label.txt', 'w')
    for label in labels:
        random.shuffle(label_partition_dict[label])
        train_num, val_num, test_num = label_partition_num[label]
        train_lines = array(label_partition_dict[label])[:train_num]
        val_lines   = array(label_partition_dict[label])[train_num:train_num+val_num]
        test_lines  = array(label_partition_dict[label])[train_num+val_num:]
        fo_train.writelines(train_lines)
        fo_val.writelines(val_lines)
        fo_test.writelines(test_lines) 
    fo_train.close()
    fo_val.close()
    fo_test.close()       

def output_results(label_num_dict, output_csv):
    label_set = array(label_num_dict.keys(), dtype = np.int)
    sample_num = label_num_dict.values()
    df = DataFrame({'label':label_set, 'sample_num':sample_num})
    df = df[['label', 'sample_num']]
    df.to_csv(output_csv,  index = False)

    plt.bar(label_set, sample_num, color = "blue")
    plt.xlabel("label")
    plt.ylabel("sample amount")
    plt.title("sample amount of label")
    plt.xlim(0, max(label_set)+2)
    plt.ylim(0, max(sample_num) + 5)
    plt.savefig(output_csv.replace('csv','jpg'), dpi=300)
    plt.close()

def get_args():
    parser = argparse.ArgumentParser(description='get style sample informations') 
    parser.add_argument('-i', dest='input_txt',
        help='list_category_img.txt', default=None, type=str) 
    parser.add_argument('-o', dest='output_csv',
        help='category_results.csv', default=None, type=str)
    parser.add_argument('-d', dest='output_dir',
        help='output_partition_dir', default=None, type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = get_args()
    input_txt = args.input_txt
    output_csv = args.output_csv
    output_dir = args.output_dir
    
    tic = time.clock()
    print 'get label_num_dict, begin...'
    label_num_dict = get_category_label_num_dict(input_txt)
    print 'get label_num_dict, done'
    print 'get partition_datasets, begin...'
    get_partition_datasets(label_num_dict, input_txt, output_dir)
    print 'get partition_datasets, done'
    toc = time.clock()
    print 'running time:{} seconds'.format(toc-tic)

    output_results(label_num_dict, output_csv)
