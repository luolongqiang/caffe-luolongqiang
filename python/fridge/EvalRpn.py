#*****************************
#date:2016-07-26
#author: luolongqiang
#*****************************

import os, sys, cv2, copy, time, argparse
import numpy as np
from numpy import array
from math import sqrt
from pandas import DataFrame
from multiprocessing import Pool, cpu_count
from EvalFinal import GetFileList, IsSubString, GetCategories, GetDistMatrixByArea, GetBoxIoU, GetMatchedLabels 

def GetLimitedLabels(input_txt, categories, level = 2, justword1 = True, isall = False):
    fi = open(input_txt,'r')
    labels = []
    while True:
        s = fi.readline()
        if not s:
            break
        obj = s.strip().split(' ')
        name = obj[0]
        if justword1 == True:
            if '_' in obj[0]:        
                name = obj[0].split('_')[0]         
        x = float(obj[1])
        y = float(obj[2])
        z = float(obj[1]) + float(obj[3])
        w = float(obj[2]) + float(obj[4])
        if isall == False:       
            if name in categories: 
                if level == 1:
                    if '-hard' or '-l2' or '-l1' not in obj[0]:
                        labels.append([name, x, y, z, w])
                elif level == 2:
                    if '-hard' or '-l2' not  in obj[0]:
                        labels.append([name, x, y, z, w])
                elif level == 3:
                    if '-hard' not in obj[0]:
                        labels.append([name, x, y, z, w])
                elif level == 4:
                    labels.append([name, x, y, z, w])
                else:
                    print("ParameterError: 'level' = 1, 2, 3, 4")
                    sys.exit()
        else:
            if name in categories or 'hard' in obj[0]:
                labels.append([name, x, y, z, w])
    fi.close()
    return labels

def GetProposals(input_txt):
    fi = open(input_txt,'r')
    labels = []
    while True:
        s = fi.readline()
        if not s:
            break
        obj = s.strip().split(' ')
        if float(obj[4]) >= threshold:
            x = float(obj[0])
            y = float(obj[1])
            z = float(obj[0]) + float(obj[2])
            w = float(obj[1]) + float(obj[3])
            labels.append(['unknow', x, y, z, w])
    fi.close()
    return labels       

def GetClassInfs(r_labels, r_matched_labels, p_matched_labels, class_dict):
    for obj in r_labels:
        class_dict[obj[0]][0] += 1        
    for r_obj, p_obj in zip(r_matched_labels, p_matched_labels):
        class_dict[r_obj[0]][1] += 1
        class_dict[r_obj[0]][4] += GetBoxIoU(r_obj, p_obj)
    return class_dict

def Experiment(real_txt_predicted): 
    real_txt = real_txt_predicted[0]
    predicted_txt = real_txt_predicted[1]
                  
    r_labels = GetLimitedLabels(real_txt, categories,  level = 2, justword1 = True, isall = False)
    p_labels = GetProposals(predicted_txt)  
    
    n, m = len(r_labels), len(p_labels)
    dist_matrix = GetDistMatrixByArea(r_labels, p_labels, abs_distance)
    r_matched_labels, p_matched_labels = \
        GetMatchedLabels(copy.copy(r_labels), copy.copy(p_labels), dist_matrix, [], [], unIoU)    
    k = len(r_matched_labels)
    IoU_list = [GetBoxIoU(r_obj, p_obj) for r_obj, p_obj in zip(r_matched_labels, p_matched_labels)]
    mIoU = np.mean(IoU_list)
    class_dict = {ele:np.zeros(5) for ele in categories}
    class_dict = GetClassInfs(r_labels, r_matched_labels, p_matched_labels, class_dict)                  
    return [n, m, k, mIoU, class_dict]

def GetClassInformationResults(results):
    class_infs = {ele:np.zeros(5) for ele in categories}
    for ele in categories:
        class_infs[ele] += sum([r[-1][ele] for r in results])
        if class_infs[ele][0] != 0:
            class_infs[ele][2] = class_infs[ele][1]*1./class_infs[ele][0]  
        if class_infs[ele][1] != 0:
            class_infs[ele][3] = class_infs[ele][4]*1./class_infs[ele][1]
    df = {'categories':categories}
    df_header = ['real_number','matched_box_number','recall', 'average_IoU']   
    for i in range(len(df_header)):
        df[df_header[i]] = [class_infs[ele][i] for ele in categories]
    df = DataFrame(df)
    df = df[['categories']+df_header]
    return df

def GetEvaluatePerformance(results): 
    n = float(sum([r[0] for r in results]))
    m = float(sum([r[1] for r in results]))
    k = float(sum([r[2] for r in results]))
    total_IoU = float(sum([r[3] for r in results]))
    num = len(results)
    ave_IoU = total_IoU/num

    precision, recall, F_score = 0., 0., 0. 
    if n != 0:
       recall = k*1./n
    if m != 0:
       precision = k*1./m
    if precision + recall != 0:
       F_score = 2*precision*recall*1.0/(precision + recall)
    return int(n/num), int(m/num), int(k/num), precision, recall, F_score, ave_IoU          
        
def GetArgs():
    parser = argparse.ArgumentParser(description='Evaluate the performance for RPN')
    parser.add_argument('-t', dest='threshold', \
        help='threshold for selecting bbox', default=0.5, type=float)
    parser.add_argument('-d', dest='abs_distance', \
        help='absolute condition for matching bbox', default=np.inf, type=int)
    parser.add_argument('-i', dest='IoU', \
        help='relative condition for matching bbox', default=0.5, type=float)
    parser.add_argument('-cfg', dest='cfg_EvalRpn', \
        help='config file for dataset', default=None, type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args
 
if __name__=="__main__":
    
    tic = time.clock()

    global threshold, abs_distance, unIoU, categories
    
    if '-cfg' in sys.argv:
        args = GetArgs()
        threshold = args.threshold
        abs_distance = args.abs_distance
        unIoU = 1 - args.IoU

        conf = {}
        with open(args.cfg_EvalRpn, 'r') as f:
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
        threshold = 0.5
        abs_distance = np.inf
        unIoU = 0.5

        if '-c' in sys.argv:
            categories = GetCategories(sys.argv[sys.argv.index('-c')+1])    
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

    if '-t' in sys.argv:
        pool = Pool(int(cpu_count()*7/8))
        results = pool.map(Experiment, zip(real_txt_list, predicted_txt_list))
        df_class_infs = GetClassInformationResults(results)
        ave_n, ave_m, ave_k, precision, recall, F_score, ave_IoU = GetEvaluatePerformance(results)
        print("******************************************************************************************")
        print(df_class_infs)
        if '-cfg' in sys.argv:
            df_class_infs.to_csv(conf['output_path'] + '/ClassInfsWithThreshold_' + str(threshold) + '.csv', index = False)
        else:
            df_class_infs.to_csv('ClassInfsWithThreshold_' + str(threshold) + '.csv', index = False)
        print("******************************************************************************************")
        print("mean_real_num:{:.3f}, mean_predicted_num:{:.3f}, mean_mathched_box_num:{:.3f}".format(ave_n, ave_m, ave_k))
        print("precision:{:.3f}, recall:{:.3f}, F_score:{:.3f}, average_IoU:{:.3f}".format(precision, recall, F_score, ave_IoU))
        print("******************************************************************************************") 
    else:
        if '-cfg' in sys.argv:
            threshold_list = [float(ele) for ele in conf['threshold_list'].split(',')]   
        else:
            threshold_list = np.arange(0.0, 1.0, 0.1)

        all_results = []
        for threshold in threshold_list:
            pool = Pool(int(cpu_count()*7/8))  
            results = pool.map(Experiment, zip(real_txt_list, predicted_txt_list))
            ave_n, ave_m, ave_k, precision, recall, F_score, ave_IoU = GetEvaluatePerformance(results)
            all_results.append([threshold, ave_n, ave_m, ave_k, precision, recall, F_score, ave_IoU])
       
        all_results = array(all_results)
        df = DataFrame({'threshold':all_results[:,0],\
                        'mean_real_number':all_results[:,1],\
                        'mean_predicted_number':all_results[:,2],\
                        'mean_matched_bbox_number':all_results[:,3],\
                        'precision':all_results[:,4],\
                        'recall':all_results[:,5],\
                        'F_score':all_results[:,6],\
                        'average_IoU': all_results[:,7]}) 
        df = df[['threshold','mean_real_number','mean_predicted_number',\
            'mean_matched_bbox_number','precision','recall','F_score', 'average_IoU']]
        if '-cfg' in sys.argv:
            df.to_csv(conf['output_path'] + '/ResultsWithThreshold.csv', index = False)
        else:
            df.to_csv('ResultsWithThreshold.csv', index = False)
        print(df)
    toc = time.clock()
    print('running time:{:.3f} seconds'.format(toc - tic))

