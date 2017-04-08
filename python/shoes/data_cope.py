# -*- coding:UTF-8 -*-
import numpy as np
from numpy import array
import os, sys, time, shutil, random

# python python/shoes/data_cope.py

if __name__ == "__main__":

    tic = time.time()

    dir1 = 'data/shoes/train.txt'
    dir2 = 'data/shoes/test.txt'
    dir3 = 'data/shoes/img.txt'
    
    list1 = open(dir1).read().splitlines()
    list2 = open(dir2).read().splitlines()
    list3 = open(dir3).read().splitlines()

    a_list = [img+'\n' for img in list(set(list1)&set(list3))]
    b_list = [img+'\n' for img in list(set(list2)&set(list3))]
    c_list = [img+'\n' for img in list(set(list3)-set(list1)-set(list2))]
    print len(a_list), len(b_list), len(c_list)

    c1_list = random.sample(c_list, int(len(c_list)*0.9))
    c2_list = list(set(c_list)-set(c1_list))
    open('data/shoes/img_train.txt', 'w').writelines(a_list+c1_list)
    open('data/shoes/img_test.txt', 'w').writelines(b_list+c2_list)
    
    toc = time.time()
    print 'running time: {} seconds'.format(toc-tic)