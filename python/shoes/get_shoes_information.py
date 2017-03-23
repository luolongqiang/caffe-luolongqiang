import numpy as np
from numpy import array
import matplotlib.pyplot as plt
from collections import Counter
from pandas import DataFrame
import os, sys, time, argparse, random

# python python/shoes/get_shoes_information.py -i data/shoes/img -o data/shoes/shoes_count.csv -p data/shoes

def get_shoes_information(img_dir, output_csv, parti_dir):     
    img_list = os.listdir(img_dir)
    label_imglist_dict = {label:[] for label in range(1,11)}
    label_list = []
    for img_file in img_list:
        label = img_file.split('_')[0]
        line = img_dir + '/' + img_file + ' ' + str(int(label)-1) + '\n'
        label_imglist_dict[int(label)].append(line)
        label_list.append(int(label))

    label_num_dict = dict(Counter(label_list))
    if output_csv:
        df = DataFrame({'label':label_num_dict.keys(), 'num':label_num_dict.values()})
        df = df[['label', 'num']]
        df.to_csv(output_csv, index=False)

    fo_train = open(parti_dir + '/shoes_train_label.txt', 'w')
    fo_val   = open(parti_dir + '/shoes_val_label.txt',   'w')
    fo_test  = open(parti_dir + '/shoes_test_label.txt',  'w')
    for label in label_num_dict.keys():
        random.shuffle(label_imglist_dict[label])
        total_num = label_num_dict[label]
        print label, total_num
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
        train_lines = array(label_imglist_dict[label])[:train_num]
        val_lines   = array(label_imglist_dict[label])[train_num:train_num+val_num]
        test_lines  = array(label_imglist_dict[label])[train_num+val_num:]
        fo_train.writelines(train_lines)
        fo_val.writelines(val_lines)
        fo_test.writelines(test_lines) 
    fo_train.close()
    fo_val.close()
    fo_test.close()

def get_args():
    parser = argparse.ArgumentParser(description='get style sample informations') 
    parser.add_argument('-i', dest='img_dir',
        help='shoes_img_dir', default=None, type=str) 
    parser.add_argument('-o', dest='output_csv',
        help='shoes_count.csv', default=None, type=str)
    parser.add_argument('-p', dest='parti_dir',
        help='partition_dir', default=None, type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = get_args()
    img_dir    = args.img_dir
    output_csv = args.output_csv
    parti_dir  = args.parti_dir
    
    tic = time.time()
    get_shoes_information(img_dir, output_csv, parti_dir)
    toc = time.time()
    print 'running time:{} seconds'.format(toc-tic)

