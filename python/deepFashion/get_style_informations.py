import numpy as np
from numpy import array
import matplotlib.pyplot as plt
from collections import Counter
from pandas import DataFrame
import os, sys, time, argparse

# python python/deepFashion/get_style_informations.py -i data/deepFashion/style_train_val_multilabel.txt -s data/deepFashion/style_name_list.txt -o data/deepFashion/others/style_train_val_informations.csv -c data/deepFashion/others/count_style_train_val_result.csv 

def get_style_sample_informations(multilabel_txt):     
    style_num = 230
    multilabels = np.zeros(style_num)
    multilabels_1_num_list = []
    i = 1
    with open(multilabel_txt, 'r') as fi:
        for line in fi: 
            print i
            temp = array(map(int, line.strip().split()[1:]))
            multilabels += temp
            multilabels_1_num_list.append(sum(temp))
            i += 1
    fi.close()
    multilabels_1_num_dict = dict(Counter(multilabels_1_num_list))
    return multilabels, multilabels_1_num_dict

def output_informations(multilabels, style_name_list_txt, style_informations_csv):
    style_name_list = []
    with open(style_name_list_txt, 'r') as fi:
        for i, line in enumerate(list(fi)):
            style_name = line[:line.index(str(i))].strip()
            style_name_list.append(style_name)
    df = DataFrame({'style_name':array(style_name_list), 'amount':multilabels})
    df = df[['style_name', 'amount']]
    df.to_csv(style_informations_csv, index = False)

    num = len(multilabels)
    x = np.arange(num)
    y = multilabels
    plt.plot(x, y, color="red", linewidth=2)
    plt.xlabel("label_id")
    plt.ylabel("sample amount")
    plt.title("sample amount of label_1 in 230 styles")
    plt.xlim(0, max(x)+1)
    plt.ylim(0, max(y) + 5)
    plt.savefig(style_informations_csv.replace('csv','jpg'), dpi=300)
    plt.close()

def output_results(multilabels_1_num_dict, count_results_csv):
    label_1_num = multilabels_1_num_dict.keys()
    sample_num = multilabels_1_num_dict.values()
    df = DataFrame({'label_1_num':label_1_num, 'sample_num':sample_num})
    df = df[['label_1_num', 'sample_num']]
    df.to_csv(count_results_csv,  index = False)

    plt.bar(label_1_num, sample_num, color = "blue")
    plt.xlabel("label_1_num")
    plt.ylabel("sample amount")
    plt.title("sample amount of label_1_num")
    plt.xlim(0, max(label_1_num))
    plt.ylim(0, max(sample_num) + 5)
    plt.savefig(count_results_csv.replace('csv','jpg'), dpi=300)
    plt.close()

def get_args():
    parser = argparse.ArgumentParser(description='get style sample informations') 
    parser.add_argument('-i', dest='input_txt',
        help='multilabel.txt', default=None, type=str)
    parser.add_argument('-s', dest='style_txt',
        help='style_name_list.txt', default=None, type=str)    
    parser.add_argument('-o', dest='output_csv',
        help='style_informations.csv', default=None, type=str)
    parser.add_argument('-c', dest='count_csv',
        help='count_results.csv', default=None, type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = get_args()
    multilabel_txt = args.input_txt
    style_name_list_txt = args.style_txt
    style_informations_csv = args.output_csv
    count_results_csv = args.count_csv
    
    tic = time.clock()
    multilabels, multilabels_1_num_dict = get_style_sample_informations(multilabel_txt)
    output_informations(multilabels, style_name_list_txt, style_informations_csv)
    output_results(multilabels_1_num_dict, count_results_csv)
    toc = time.clock()
    print 'running time:{} seconds'.format(toc-tic)
