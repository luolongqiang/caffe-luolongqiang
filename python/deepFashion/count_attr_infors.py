import numpy as np
from numpy import array
import matplotlib.pyplot as plt
from collections import Counter
from pandas import DataFrame
import os, sys, time, argparse

# python python/deepFashion/count_attr_infors.py -m data/deepFashion/Anno/list_attr_img.txt -n data/deepFashion/Anno/list_attr_cloth.txt -d data/deepFashion/others/attr_distr_results.csv -c data/deepFashion/others/attr_count_results.csv
# scp /home/luolongqiang/deepdraw/caffe/python/deepFashion/count_attr_infors.py luolongqiang@192.168.1.182:/home/luolongqiang/deepdraw/caffe/python/deepFashion/count_attr_infors.py

def get_attr_infors(multilabel_txt):     
    multilabels = np.zeros(1000)
    multilabels_1_num_list = []
    with open(multilabel_txt, 'r') as fi:
        for i, line in enumerate(list(fi)[2:]):
            print i 
            temp = array(map(int, line.strip().split()[1:]))==1
            multilabels += temp==1
            multilabels_1_num_list.append(sum(temp))
    fi.close()
    multilabels_1_num_dict = dict(Counter(multilabels_1_num_list))
    print multilabels, multilabels_1_num_dict
    return multilabels, multilabels_1_num_dict

def output_distr(multilabels, name_txt, attr_distr_csv):
    fi_list = open(name_txt, 'r').read().splitlines()[2:]
    name_list = [line.strip()[:-1].strip() for line in fi_list]
    df = DataFrame({'name':array(name_list), 'num':multilabels})
    df = df[['name', 'num']]
    df.to_csv(attr_distr_csv, index = False)

    num = len(multilabels)
    x = np.arange(num)
    y = multilabels
    plt.plot(x, y, color="red", linewidth=2)
    plt.xlabel("label_id")
    plt.ylabel("label distribution")
    plt.title("label_distribution vs label_id")
    plt.xlim(0, max(x) + 1)
    plt.ylim(0, max(y) + 5)
    plt.savefig(attr_distr_csv.replace('csv','jpg'), dpi=300)
    plt.close()

def output_results(multilabels_1_num_dict, attr_count_csv):
    label_1_num = multilabels_1_num_dict.keys()
    sample_num = multilabels_1_num_dict.values()
    df = DataFrame({'label_1_num':label_1_num, 'sample_num':sample_num})
    df = df[['label_1_num', 'sample_num']]
    df.to_csv(attr_count_csv,  index = False)

    plt.bar(label_1_num, sample_num, color = "blue")
    plt.xlabel("label_1_num")
    plt.ylabel("sample_num")
    plt.title("sample_num vs label_1_num")
    plt.xlim(0, max(label_1_num))
    plt.ylim(0, max(sample_num) + 5)
    plt.savefig(attr_count_csv.replace('csv','jpg'), dpi=300)
    plt.close()

def get_args():
    parser = argparse.ArgumentParser(description='count attribution informations') 
    parser.add_argument('-m', dest='multilabel_txt',
        help='list_attr_img.txt', default=None, type=str)  
    parser.add_argument('-n', dest='name_txt',
        help='list_attr_cloth.txt', default=None, type=str)  
    parser.add_argument('-d', dest='distr_csv',
        help='attr_distr_results.csv', default=None, type=str)
    parser.add_argument('-c', dest='count_csv',
        help='attr_count_results.csv', default=None, type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = get_args()
    multilabel_txt = args.multilabel_txt
    name_txt       = args.name_txt
    attr_distr_csv = args.distr_csv
    attr_count_csv = args.count_csv
    
    tic = time.clock()
    multilabels, multilabels_1_num_dict = get_attr_infors(multilabel_txt)
    output_distr(multilabels, name_txt, attr_distr_csv)
    output_results(multilabels_1_num_dict, attr_count_csv)
    toc = time.clock()
    print 'running time:{} seconds'.format(toc-tic)
