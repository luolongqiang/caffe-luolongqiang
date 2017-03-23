import os, sys, time
import argparse
import numpy as np
from numpy import array

# python python/deepFashion/get_attr_multilabel.py -m data/deepFashion/Anno/list_attr_img.txt -p data/deepFashion/Eval/list_eval_partition.txt -o data/deepFashion
# scp /home/luolongqiang/deepdraw/caffe/python/deepFashion/get_attr_multilabel.py luolongqiang@192.168.1.182:/home/luolongqiang/deepdraw/caffe/python/deepFashion/get_attr_multilabel.py

def get_attr_multilabel(multilabel_txt, partition_txt, output_dir):      
    partition_dict ={}
    with open(partition_txt, 'r') as fi:
        for line in list(fi)[2:]:
             line_list = line.strip().split()
             partition_dict[line_list[0]]=line_list[-1]
    fi.close()

    fo_train = open(output_dir+'/attr_train_multilabel.txt', 'w')
    fo_val   = open(output_dir+'/attr_val_multilabel.txt', 'w')
    fo_test  = open(output_dir+'/attr_test_multilabel.txt', 'w')
    
    i = 0
    with open(multilabel_txt, 'r') as fi:
        for line in list(fi)[2:]:
            line_list = line.strip().split()
            img_file_name = line_list[0]
            style_multilabel_array = array(line_list[1:])=='1'
            if sum(style_multilabel_array)==0:
                i += 1
                continue
            style_multilabel_str = ''
            for label in style_multilabel_array:
               style_multilabel_str += ' ' + str(int(label)) 
            new_line = 'data/deepFashion/' + img_file_name + style_multilabel_str + '\n'
            if partition_dict[img_file_name] == 'train':
                fo_train.write(new_line)
            if partition_dict[img_file_name] == 'val':
                fo_val.write(new_line)
            if partition_dict[img_file_name] == 'test':
                fo_test.write(new_line)
       
    fi.close()
    fo_train.close()
    fo_val.close()
    fo_test.close()

def get_args():
    parser = argparse.ArgumentParser(description='get attr train, validation and test multilabels') 
    parser.add_argument('-m', dest='multilabel',
        help='list_attr_img.txt', default=None, type=str)
    parser.add_argument('-p', dest='partition',
        help='list_eval_partition.txt', default=None, type=str)
    parser.add_argument('-o', dest='output',
        help='attr_multilabel_dir', default=None, type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = get_args()  
    multilabel_txt = args.multilabel
    partition_txt = args.partition
    output_dir = args.output

    tic = time.clock()
    get_attr_multilabel(multilabel_txt, partition_txt, output_dir)
    toc = time.clock()
    print 'running time:{} seconds'.format(toc-tic)
