import numpy as np
from numpy import array
from collections import Counter
import os, sys, time, argparse, random

# python python/shoes/temp.py -i data/shoes/taobao -o data/shoes/taobao_test_label.txt
# scp /home/luolongqiang/deepdraw/shoes/taobao_test_label.txt luolongqiang@192.168.1.182:/home/luolongqiang/deepdraw/shoes/taobao_test_label.txt

def get_shoes_test(img_dir, output_txt):     
    img_list = os.listdir(img_dir)
    fo = open(output_txt, 'w')
    for img_file in img_list:
        label = img_file.split('.')[0].split('-')[-1]
        line = img_dir + '/' + img_file + ' ' + str(int(label)-1) + '\n'
        fo.write(line)
    fo.close()

def get_args():
    parser = argparse.ArgumentParser(description='get style sample informations') 
    parser.add_argument('-i', dest='img_dir',
        help='shoes img dir', default=None, type=str) 
    parser.add_argument('-o', dest='output_txt',
        help='taobao_test.txt', default=None, type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = get_args()
    
    img_dir    = args.img_dir
    output_txt = args.output_txt
    
    tic = time.time()
    get_shoes_test(img_dir, output_txt)
    toc = time.time()
    print 'running time:{} seconds'.format(toc-tic)