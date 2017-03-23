#*****************************
#date:2016-07-26
#author:luolongqiang
#*****************************

import os, sys, time, copy, argparse
import numpy as np
from numpy import array
from math import sqrt
from pandas import DataFrame
from multiprocessing import Pool, cpu_count
from EvalFinal import GetFileList, IsSubString, GetCategories, \
    GetDistMatrixByArea, GetBoxIoU, GetMatchedLabels

def GetLabelsWithThreshold(input_txt, name, threshold, real_sample):
    fi = open(input_txt,'r')
    labels = []
    while True:
        s = fi.readline()
        if not s:
            break
        obj = s.strip().split(' ')
        x = float(obj[1])
        y = float(obj[2])
        z = float(obj[1]) + float(obj[3])
        w = float(obj[2]) + float(obj[4])
        if name in obj[0]:
            if real_sample == True:
                labels.append([name, x, y, z, w]) 
            else:
                if float(obj[5]) >= threshold:
                    labels.append([name, x, y, z, w])
    fi.close()                       
    return labels

def Experiment(name):
    maxp, maxr, maxf = 0., 0., 0.
    optimal_threshold = 0. 
    for threshold in np.arange(0.1, 1., 0.01):
        N, M, K = 0., 0., 0.
        for i in range(len(real_txt_list)):
            real_txt = real_txt_list[i]
            predicted_txt = predicted_txt_list[i]
            r_labels = GetLabelsWithThreshold(real_txt, name, threshold, True)  
            p_labels = GetLabelsWithThreshold(predicted_txt, name, threshold, False)            
            N += len(r_labels)
            M += len(p_labels)
            dist_matrix = GetDistMatrixByArea(r_labels, p_labels, abs_distance)
            r_matched_labels, p_matched_labels = \
               GetMatchedLabels(copy.copy(r_labels), copy.copy(p_labels), dist_matrix, [], [], unIoU)                       
            K += len(r_matched_labels)        
        precision, recall, F_score = 0., 0., 0.
        if M != 0:     
            precision = K/M
        if N != 0:    
            recall = K/N
        if precision + recall != 0:
            F_score = 2*precision*recall/(precision+recall)
        if maxf < F_score:
            maxp = precision
            maxr = recall
            maxf = F_score
            optimal_threshold = threshold
    if maxp + maxr + maxf == 0:
        optimal_threshold = 1
    print("*****************************************************************************************")
    print("name:{:s},optimal_threshold:{:.3f}, precision:{:.3f}; recall:{:.3f}; F_score:{:.3f}"\
        .format(name, optimal_threshold, maxp, maxr, maxf))
    print("*****************************************************************************************")
    return [name, optimal_threshold, maxp, maxr, maxf]
        
def GetArgs():
    parser = argparse.ArgumentParser(description='Get optimal threshold for each category')
    parser.add_argument('-d', dest='abs_distance', \
        help='absolute condition for matching bbox', default=np.inf, type=int)
    parser.add_argument('-i', dest='IoU', \
        help='relative condition for matching bbox', default=0.5, type=float)
    parser.add_argument('-cfg', dest='cfg_GetOptimalThreshold', \
        help='config file for dataset', default=None, type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

if __name__=="__main__":   
    tic = time.clock()
        
    global categories, abs_distance, unIoU, real_txt_list, predicted_txt_list

    if '-cfg' in sys.argv:   
        args = GetArgs()
        abs_distance = args.abs_distance
        unIoU = 1 - args.IoU  
   
        conf = {}
        with open(args.cfg_GetOptimalThreshold, 'r') as f:
            for line in f:
                infos = line.strip().split('=')
                conf[infos[0]] = infos[1]

        categories = conf['cls_names'].split(',')    

        real_txt_list = []
        real_root = conf['real_root']
        for data_dir in conf['real_dir'].split(','):
            complete_pth = os.path.join(real_root, data_dir)
            real_txt_list += GetFileList(complete_pth, ['.txt'])

        predicted_txt_list = []
        predicted_root = conf['predicted_root']
        for data_dir in conf['predicted_dir'].split(','):
            complete_pth = os.path.join(predicted_root, data_dir)
            predicted_txt_list += GetFileList(complete_pth, ['.txt'])
    else:
        if len(sys.argv) == 1:
           args = GetArgs()

        categories = ['apple','beer','broccoli','chineseCabbage','cucumber','egg',\
                    'eggplant','grape','ham','milk','onion','orange','papaya','pear',\
                    'potato','radish','strawberry','tomato','watermelon','whiteradish']            
        abs_distance = np.inf
        unIoU = 0.5
 
        if '-c' in sys.argv:
            categories = GetCategories(sys.argv[sys.argv.index('-c')+1])    
        if '-t1' in sys.argv:
            abs_distance = float(sys.argv[sys.argv.index('-t1')+1])
        if '-t2' in sys.argv:
            unIoU = 1- float(sys.argv[sys.argv.index('-t2')+1])
       
        real_file = GetFileList(sys.argv[1], [])
        predicted_file = GetFileList(sys.argv[2], [])
        real_txt_list = []
        predicted_txt_list = []
        for pth in real_file:
            real_txt_list += GetFileList(pth + '/txt',['.txt'])
        for pth in predicted_file:     
            predicted_txt_list += GetFileList(pth,['.txt'])

    pool = Pool(int(cpu_count()*7/8))
    results = pool.map(Experiment, categories)
    
    results = array(results)   
    df = DataFrame({'categories':results[:,0],\
                         'optimal_threshold':results[:,1],\
                         'precision':results[:,2],\
                         'recall':results[:,3],\
                         'F_score':results[:,4]})
    df = df[['categories','optimal_threshold', 'precision', 'recall', 'F_score']]
    if '-cfg' in sys.argv:
        df.to_csv(conf['output_path'] + '/OptimalThreshold.csv', index = False)
    else:
        df.to_csv('OptimalThreshold.csv', index=False)

    toc = time.clock()   
    print('running time:{:.3f} seconds'.format(toc-tic))
    

            
