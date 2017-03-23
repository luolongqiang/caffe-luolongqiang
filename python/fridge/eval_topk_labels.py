#*****************************
#date:2016-07-26
#author:luolongqiang
#*****************************

import os, sys, cv2, copy, time, argparse
import numpy as np
from numpy import array
from math import sqrt
from pandas import DataFrame
from multiprocessing import Pool, cpu_count
from EvalRpn import GetLimitedLabels
from EvalFinal import GetFileList, IsSubString, GetCategories, GetDistMatrixByArea, 、
    GetBoxIoU, GetMatchedLabels, GetMissingLabels, PerformanceToCsv

def GetPredictedLabels(input_txt):
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
        labels.append([obj[0], x, y, z, w])
    fi.close()
    return labels           

def EvaluatePerformance(n, m, r_matched_labels, p_matched_labels, r_missing_matched_labels, p_missing_matched_labels): 
    if len(r_matched_labels) == 0:
        return m, 0., 0., 0., 0., 0., 1., 1., 1., r_matched_labels, p_matched_labels
    r_names = array(r_matched_labels)[:,0]
    p_names = array(p_matched_labels)[:,0]
    k = len(r_names)

    l = 0
    if topk == 1:
        for i in range(len(r_names)):
            if r_names[i]==p_names[i].split(',')[0]:
               l += 1
    elif topk == 2:
        for i in range(len(r_names)): 
            if r_names[i] in p_names[i].split(',')[0:2]:
                l += 1
    else:
        for i in range(len(r_names)):
            if r_names[i] in p_names[i].split(','):
                l += 1

    j = 0 
    if len(r_missing_matched_labels) > 0:
        for i in range(len(r_missing_matched_labels)):
            r_obj = r_missing_matched_labels[i]
            p_obj = p_missing_matched_labels[i]
            if 'hard' in r_obj[0]:
                j = j + 1
                r_matched_labels.append([r_obj[0], r_obj[1], r_obj[2], r_obj[3], r_obj[4]])
                p_matched_labels.append([p_obj[0], p_obj[1], p_obj[2], p_obj[3], p_obj[4]])

    m = m-j
    precision, recall = l*1./m, l*1./n
    F_score = 2*precision*recall*1.0/(precision + recall)
    error_rate = 1-l*1./k
    missing_rate = (n-k)*1./n
    over_detected_rate = (m-k)*1./m 
    return m, k, l, precision, recall, F_score, error_rate, missing_rate, over_detected_rate, r_matched_labels, p_matched_labels        

def Experiment(real_txt_predicted): 
    real_txt = real_txt_predicted[0]
    predicted_txt = real_txt_predicted[1]
                  
    r_labels = GetLimitedLabels(real_txt, categories, level = 2, justword1 = True, isall = False)
    p_labels = GetPredictedLabels(predicted_txt)
    n, m = len(r_labels), len(p_labels)

    dist_matrix = GetDistMatrixByArea(r_labels, p_labels, abs_distance)
    r_matched_labels, p_matched_labels = GetMatchedLabels(copy.copy(r_labels), copy.copy(p_labels), dist_matrix, [], [], unIoU)    
  
    r_all_labels = GetLimitedLabels(real_txt, categories, level = 0, justword1 = True, isall = True)
    p_all_labels = GetPredictedLabels(predicted_txt)   
    
    r_missing_labels = GetMissingLabels(copy.copy(r_all_labels), copy.copy(r_matched_labels))   
    p_missing_labels = GetMissingLabels(copy.copy(p_all_labels), copy.copy(p_matched_labels))  
    
    missing_dist_matrix = GetDistMatrixByArea(copy.copy(r_missing_labels), copy.copy(p_missing_labels), abs_distance)
    r_missing_matched_labels, p_missing_matched_labels = \
       GetMatchedLabels(copy.copy(r_missing_labels), copy.copy(p_missing_labels), missing_dist_matrix, [], [], unIoU)    
            
    m, k, l, precision, recall, F_score, error_rate, missing_rate, over_detected_rate, r_matched_labels, p_matched_labels = \
       EvaluatePerformance(n, m, r_matched_labels, p_matched_labels, r_missing_matched_labels, p_missing_matched_labels)       
    return [real_txt.split('/')[-1].split('.')[0], n, m, k, l, precision, recall, F_score, error_rate, missing_rate, over_detected_rate, {}]

def GetArgs():
    parser = argparse.ArgumentParser(description='Evaluate the performance for final prediction')
    parser.add_argument('-topk', dest='topk', help='a bbox has k lables', default=1, type=int)
    parser.add_argument('-d', dest='abs_distance', help='absolute condition for matching bbox', default=np.inf, type=int)
    parser.add_argument('-i', dest='IoU', help='relative condition for matching bbox', default=0.5, type=float)
    parser.add_argument('-cfg', dest='cfg_EvalFinal', help='config file for dataset', default=None, type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args
        
if __name__=="__main__":
    
    tic = time.clock()
    
    global topk, abs_distance, unIoU, categories, conf
   
    if '-cfg' in sys.argv:
        args = GetArgs()
        topk = args.topk
        abs_distance = args.abs_distance
        unIoU = 1 - args.IoU
       
        conf = {}
        with open(args.cfg_EvalFinal, 'r') as f:
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
                      'eggplant','grape','ham','milk','onion','orange','bitter','pear',\
                      'potato','radish','strawberry','tomato','watermelon','whiteradish']
        topk = 1
        abs_distance = np.inf
        unIoU = 0.5

        if '-c' in sys.argv:
            categories = GetCategories(sys.argv[sys.argv.index('-c')+1])    
        if '-topk' in sys.argv:
            topk = int(sys.argv[sys.argv.index('-topk')+1])
        if '-d' in sys.argv:
            abs_distance = float(sys.argv[sys.argv.index('-d')+1])
        if '-i' in sys.argv:
            unIoU = 1- float(sys.argv[sys.argv.index('-i')+1])

        real_file = GetFileList(sys.argv[1], [])
        predicted_file = GetFileList(sys.argv[2], [])
        real_txt_list = []
        predicted_txt_list = []
        for pth in real_file:
            real_txt_list += GetFileList(pth + '/txt',['.txt'])
        for pth in predicted_file:     
            predicted_txt_list += GetFileList(pth,['.txt'])

    pool = Pool(int(cpu_count()*7/8))
    results = pool.map(Experiment, zip(real_txt_list, predicted_txt_list))
    precision, recall, F_score, error_rate, missing_rate, over_detected_rate = PerformanceToCsv(results) 

    print("************************************************************************************************************")
    print("precision:{:.3f}; recall:{:.3f}; F_score:{:.3f}; error_rate:{:.3f}; missing_rate:{:.3f}; over_detected_rate:{:.3f}"\
          .format(precision, recall, F_score, error_rate, missing_rate, over_detected_rate))
    print("************************************************************************************************************") 
   
    toc = time.clock()
    print('running time: {:.3f} seconds'.format(toc-tic))
    
