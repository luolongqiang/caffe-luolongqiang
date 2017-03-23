import os, sys, time
import argparse
import numpy as np
from numpy import array

def get_style_multilabel(type_txt, multilabel_txt, partition_txt, output_dir):      
  # k = 0
    style_index_list = []
    with open(type_txt, 'r') as fi:
        for i, line in enumerate(list(fi)[2:]):
            line = line.strip()
            if int(line[-1]) == 5:
              # print line[:-1].strip(), k, i
                style_index_list.append(i)
              # k += 1
    fi.close()

    partition_dict ={}
    with open(partition_txt, 'r') as fi:
        for line in list(fi)[2:]:
             line_list = line.strip().split()
             partition_dict[line_list[0]]=line_list[-1]
    fi.close()

    fo_train = open(output_dir+'/style_train_multilabel.txt', 'w')
    fo_val   = open(output_dir+'/style_val_multilabel.txt', 'w')
    fo_test  = open(output_dir+'/style_test_multilabel.txt', 'w')
    
    i = 1
    with open(multilabel_txt, 'r') as fi:
        for line in list(fi)[2:]:
            print i
            line_list = line.split()
            img_file_name = line_list[0]
            style_multilabel_array = array(line_list[1:])[style_index_list]=='1'
            style_multilabel_str = str(int(style_multilabel_array[0]))
            for label in style_multilabel_array[1:]:
               style_multilabel_str += ' ' + str(int(label)) 
            new_line = output_dir + '/' + img_file_name + ' ' + style_multilabel_str + '\n'
            if partition_dict[img_file_name] == 'train':
                fo_train.write(new_line)
            if partition_dict[img_file_name] == 'val':
                fo_val.write(new_line)
            if partition_dict[img_file_name] == 'test':
                fo_test.write(new_line)
            i += 1           
    fi.close()
    fo_train.close()
    fo_val.close()
    fo_test.close()

def get_args():
    parser = argparse.ArgumentParser(description='get train, validation and test style multilabels') 
    parser.add_argument('-t', dest='type',
        help='list_attr_cloth.txt', default=None, type=str)
    parser.add_argument('-m', dest='multilabel',
        help='list_attr_img.txt', default=None, type=str)
    parser.add_argument('-p', dest='partition',
        help='list_eval_partition.txt', default=None, type=str)
    parser.add_argument('-o', dest='output',
        help='style_multilabel_dir', default=None, type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = get_args()
    type_txt = args.type   
    multilabel_txt = args.multilabel
    partition_txt = args.partition
    output_dir = args.output

    tic = time.clock()
    get_style_multilabel(type_txt, multilabel_txt, partition_txt, output_dir)
    toc = time.clock()
    print 'running time:{} seconds'.format(toc-tic)
