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

def GetFileList(FindPath, FlagStr=[]):     
    FileList = []  
    FileNames = os.listdir(FindPath)  
    if len(FileNames) > 0:  
       for fn in FileNames:  
           if len(FlagStr) > 0:  
               if IsSubString(FlagStr,fn):  
                   fullfilename = os.path.join(FindPath,fn)  
                   FileList.append(fullfilename)  
           else:  
               fullfilename=os.path.join(FindPath,fn)  
               FileList.append(fullfilename)  
    if len(FileList)>0:  
        FileList.sort()    
    return FileList 
    
def IsSubString(SubStrList,Str):  
    flag=True  
    for substr in SubStrList:  
        if not (substr in Str):  
            flag=False  
    return flag

def GetCategories(categories_file):
    categories=[]
    with open(categories_file, 'r') as fi:
    	for s in fi:
            categories.append(s.strip())
    return categories

def GetBoxRange(input_txt):
    labels = []
    with open(input_txt, 'r') as fi:
    	for s in fi:
	    obj=s.strip().split(' ')
	    x = int(float(obj[1]))
	    y = int(float(obj[2]))
	    z = int(float(obj[1])) + int(float(obj[3]))
	    w = int(float(obj[2])) + int(float(obj[4]))
	    labels.append([x,y,z,w])
    labels = array(labels)  
    return [min(labels[:,0]), min(labels[:,1]), max(labels[:,2]), max(labels[:,3])]

def GetBoxIoA(box_range, axes):
    box_area = (axes[2]-axes[0]+1.0)*(axes[3]-axes[1]+1.0)
    xmax, ymax = max(box_range[0], axes[0]), max(box_range[1], axes[1])
    zmin, wmin = min(box_range[2], axes[2]), min(box_range[3], axes[3])
    w = max(zmin - xmax + 1.0, 0.0)
    h = max(wmin - ymax + 1.0, 0.0)
    inter_area = w * h
    box_IoA = 0
    if box_area != 0:
        box_IoA = inter_area*1.0/box_area
    return box_IoA

def GetLimitedLabels(input_txt, categories, box_range, isbox = 1, level = 2, justword1 = True, isall = False):
    labels = []
    with open(input_txt,'r') as fi:
    	for s in fi:
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
                        if isbox == 1:
                            box_IoA = GetBoxIoA(box_range, [x, y, z, w])
                            if box_IoA > 0.5:
                                labels.append([name, x, y, z, w])
                        else:       
                            labels.append([name, x, y, z, w])
	            elif level == 2:
	                if '-hard' or '-l2' not  in obj[0]:
                        if isbox == 1:
                            box_IoA = GetBoxIoA(box_range, [x, y, z, w])
                            if box_IoA > 0.5:
                                labels.append([name, x, y, z, w])
                        else:       
                            labels.append([name, x, y, z, w])
	            elif level == 3:
	                if '-hard' not in obj[0]:
                        if isbox == 1:
                            box_IoA = GetBoxIoA(box_range, [x, y, z, w])
                            if box_IoA > 0.5:
                                labels.append([name, x, y, z, w])
                        else:       
                            labels.append([name, x, y, z, w])
	            elif level == 4:
                    if isbox == 1:
                        box_IoA = GetBoxIoA(box_range, [x, y, z, w])
                        if box_IoA > 0.5:
                            labels.append([name, x, y, z, w])
                    else:       
                        labels.append([name, x, y, z, w])
	            else:
	                print("ParameterError: 'level' = 1, 2, 3, 4")
	                sys.exit()
	    else:
	        if name in categories or 'hard' in obj[0]:
                if isbox == 1:
                    box_IoA = GetBoxIoA(box_range, [x, y, z, w])
        	        if box_IoA > 0.5:
                        labels.append([name, x, y, z, w])
                else:       
            	    labels.append([name, x, y, z, w])
    return labels
    
def GetPerClassInformation(r_labels, p_labels, class_dict):
    for ele in r_labels:
        class_dict[ele[0]][0] += 1
    for ele in p_labels:
        class_dict[ele[0]][1] += 1
    return class_dict  

def GetDistMatrixByArea(r_labels, p_labels, abs_distance):
    n, m = len(r_labels), len(p_labels)
    dist_matrix = np.ones((n, m))
    for i in range(n):
        x1, y1 = r_labels[i][1], r_labels[i][2]
        z1, w1 = r_labels[i][3], r_labels[i][4] 
        for j in range(m):  
            x2, y2 = p_labels[j][1], p_labels[j][2]
            z2, w2 = p_labels[j][3], p_labels[j][4]
            d1 = sqrt((x1-x2)**2 + (y1-y2)**2)
            d2 = sqrt((z1-z2)**2 + (w1-w2)**2)
            if d1 < abs_distance and d2 < abs_distance:
                dist_matrix[i,j] = 1 - GetBoxIoU(r_labels[i], p_labels[j])
    return dist_matrix    

def GetBoxIoU(r_obj, p_obj):
    x1, y1, z1, w1 = r_obj[1], r_obj[2], r_obj[3], r_obj[4]
    x2, y2, z2, w2 = p_obj[1], p_obj[2], p_obj[3], p_obj[4]
    box1_area = (z1-x1+1.0)*(w1-y1+1.0)
    box2_area = (z2-x2+1.0)*(w2-y2+1.0)
    ixmin, iymin = max(x1,x2), max(y1,y2)
    ixmax, iymax = min(z1,z2), min(w1,w2)
    iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
    ih = np.maximum(iymax - iymin + 1.0, 0.0)
    inter_area = iw * ih
    union_area = box1_area + box2_area - inter_area
    iIou = 0
    if union_area != 0:
        iIoU = inter_area/union_area    
    return iIoU 

def GetMatchedLabels(r_labels, p_labels, dist_matrix, r_matched_labels, p_matched_labels, unIoU):
    if dist_matrix.shape[0] == 0 or dist_matrix.shape[1] == 0 or np.min(dist_matrix) > unIoU:
        return r_matched_labels, p_matched_labels
    else:
        min_row_index = np.argmin(dist_matrix)//dist_matrix.shape[1]
        min_col_index = np.argmin(dist_matrix) - min_row_index*dist_matrix.shape[1]
        r_matched_labels.append(r_labels.pop(min_row_index))
        p_matched_labels.append(p_labels.pop(min_col_index))
        dist_matrix = list(dist_matrix)
        dist_matrix.pop(min_row_index)   
        dist_matrix = list(np.transpose(dist_matrix))
        if len(dist_matrix)!=0:
            dist_matrix.pop(min_col_index)
        dist_matrix = np.transpose(dist_matrix)
        return GetMatchedLabels(r_labels, p_labels, dist_matrix, r_matched_labels, p_matched_labels, unIoU)        

def GetMissingLabels(all_labels, matched_labels):
    if len(matched_labels)>0:
        for ele in matched_labels:
            all_labels.remove(ele)
    return all_labels

def EvaluatePerformance(n, m, r_matched_labels, p_matched_labels, r_missing_matched_labels, p_missing_matched_labels, class_dict): 
    if len(r_matched_labels) == 0:
        return m, 0., 0., 0., 0., 0., 1., 1., 1., 0, class_dict    

    k = len(r_matched_labels)
    l = sum(array(r_matched_labels)[:,0] == array(p_matched_labels)[:,0])  

    for r_obj, p_obj in zip(r_matched_labels, p_matched_labels):
    	class_dict[r_obj[0]][2] += 1
    	class_dict[p_obj[0]][3] += 1
    	if r_obj[0] == p_obj[0]:
    		class_dict[r_obj[0]][4] += 1
    		class_dict[r_obj[0]][12] += GetBoxIoU(r_obj, p_obj)

    j = 0 
    if len(r_missing_matched_labels) > 0:
        for r_obj, p_obj in zip(r_missing_matched_labels, p_missing_matched_labels):
            if 'hard' in r_obj[0]:
                j = j + 1
                r_matched_labels.append([r_obj[0], r_obj[1], r_obj[2], r_obj[3], r_obj[4]])
                p_matched_labels.append([p_obj[0], p_obj[1], p_obj[2], p_obj[3], p_obj[4]])
                class_dict[p_obj[0]][1] -= 1
    m = m-j
    precision, recall = l*1./m, l*1./n
    F_score = 2*precision*recall*1.0/(precision + recall)
    error_rate = 1-l*1./k
    missing_rate = (n-k)*1./n
    over_detected_rate = (m-k)*1./m 
	
    IoU_list = [GetBoxIoU(r_obj, p_obj) for r_obj, p_obj in zip(r_matched_labels, p_matched_labels)]
    average_IoU = np.mean(IoU_list)

    return m, k, l, precision, recall, F_score, error_rate, missing_rate, over_detected_rate, average_IoU, class_dict             

def Experiment(real_txt_predicted): 
    real_txt = real_txt_predicted[0]
    predicted_txt = real_txt_predicted[1]
    
    box_range = GetBoxRange(real_txt)            
    r_labels = GetLimitedLabels(real_txt, categories, box_range, isbox = 0, level = 2, justword1 = True, isall = False)
    p_labels = GetLimitedLabels(predicted_txt, categories, box_range, isbox = is_box_range, level = 2, justword1 = True, isall = False)  
  
    class_dict = {ele:-np.zeros(13) for ele in categories}
    class_dict = GetPerClassInformation(r_labels, p_labels, class_dict)  
    n, m = len(r_labels), len(p_labels)

    dist_matrix = GetDistMatrixByArea(r_labels, p_labels, abs_distance)
    r_matched_labels, p_matched_labels = GetMatchedLabels(copy.copy(r_labels), copy.copy(p_labels), dist_matrix, [], [], unIoU)    
  
    r_all_labels = GetLimitedLabels(real_txt, categories, box_range, isbox = 0, level = 0, justword1 = True, isall = True)
    p_all_labels = GetLimitedLabels(predicted_txt, categories, box_range, isbox = is_box_range, level = 0, justword1 = True, isall = True)   
    
    r_missing_labels = GetMissingLabels(copy.copy(r_all_labels), copy.copy(r_matched_labels))   
    p_missing_labels = GetMissingLabels(copy.copy(p_all_labels), copy.copy(p_matched_labels))  
    
    missing_dist_matrix = GetDistMatrixByArea(copy.copy(r_missing_labels), copy.copy(p_missing_labels), abs_distance)
    r_missing_matched_labels, p_missing_matched_labels = \
       GetMatchedLabels(copy.copy(r_missing_labels), copy.copy(p_missing_labels), missing_dist_matrix, [], [], unIoU)    
            
    m, k, l, precision, recall, F_score, error_rate, missing_rate, over_detected_rate, average_IoU, class_dict = \
        EvaluatePerformance(n, m, r_matched_labels, p_matched_labels, r_missing_matched_labels, p_missing_matched_labels, class_dict)       

    return [real_txt.split('/')[-1].split('.')[0], n, m, k, l, precision, recall, F_score,\
        error_rate, missing_rate, over_detected_rate, average_IoU, class_dict]

def ClassInformationToCsv(results):
    class_infs = {ele:-np.zeros(13) for ele in categories}
    for ele in categories:
        class_infs[ele] += sum([r[-1][ele] for r in results])
        if class_infs[ele][0] != 0:
            class_infs[ele][6] = class_infs[ele][4]*1./class_infs[ele][0]
            class_infs[ele][9] = (class_infs[ele][0] - class_infs[ele][2])*1./class_infs[ele][0]
        if class_infs[ele][1] != 0:
            class_infs[ele][5] = class_infs[ele][4]*1./class_infs[ele][1]
            class_infs[ele][10] = (class_infs[ele][1] - class_infs[ele][3])*1./class_infs[ele][1]
        if class_infs[ele][2] != 0:
            class_infs[ele][8] = 1 - class_infs[ele][4]*1./class_infs[ele][2]
        if class_infs[ele][5] + class_infs[ele][6] != 0:
            class_infs[ele][7] = 2*class_infs[ele][5]*class_infs[ele][6]/(class_infs[ele][5]+class_infs[ele][6])
        if class_infs[ele][4] != 0:
        	class_infs[ele][11] = class_infs[ele][12]*1./class_infs[ele][4]

    df = {'categories':categories}
    df_header = ['real_number','predicted_number','matched_real_box_number',\
        'matched_predicted_box_number','matched_label_number','precision','recall',\
        'F_score','error_rate','missing_rate','over_detected_rate', 'average_IoU']   
    for i in range(len(df_header)):
        df[df_header[i]] = [class_infs[ele][i] for ele in categories]
    df = DataFrame(df)
    df = df[['categories']+df_header]
    if '-cfg' in sys.argv:
        df.to_csv(conf['output_path'] + '/ClassInfsOfEvaluation.csv', index=False)
    else:
        df.to_csv('ClassInfsOfEvaluation.csv', index=False)
    return df

def PerformanceToCsv(results):
    N = float(sum([r[1] for r in results]))
    M = float(sum([r[2] for r in results]))
    K = float(sum([r[3] for r in results]))
    L = float(sum([r[4] for r in results]))
    Total_IoU = sum([r[-2] for r in results])
    average_IoU = Total_IoU/len(results)
    
    precision, recall, F_score =0., 0., 0.
    error_rate, missing_rate, over_detected_rate = 1., 1., 1.
    if M != 0:
        precision = L/M
        over_detected_rate = (M-K)/M
    if N != 0:
        recall = L/N
        missing_rate = (N-K)/N
    if K != 0:
        error_rate = 1-L/K
    if precision + recall != 0:
        F_score = 2*precision*recall/(precision+recall)
  
    results.append(['total', N, M, K, L, precision, recall, F_score, error_rate,\
                               missing_rate, over_detected_rate, average_IoU, {}]) 
    results=array(results)
    df=DataFrame({'picture':results[:,0],\
                  'real_number':results[:,1],\
                  'predicted_number':results[:,2],\
                  'matched_box_number':results[:,3],\
                  'matched_label_number':results[:,4],\
                  'precision':results[:,5],\
                  'recall':results[:,6],\
                  'F_score':results[:,7],\
                  'error_rate':results[:,8],\
                  'missing_rate':results[:,9],\
                  'over_detected_rate':results[:,10],\
                  'average_IoU':results[:,11]})
    df=df[['picture','real_number','predicted_number','matched_box_number','matched_label_number',\
        'precision','recall','F_score','error_rate','missing_rate','over_detected_rate', 'average_IoU']]    
    if '-cfg' in sys.argv:
        df.to_csv(conf['output_path'] + '/ResultsOfEvaluation.csv', index=False) 
    else:
        df.to_csv('ResultsOfEvaluation.csv', index=False)
    return precision, recall, F_score, error_rate, missing_rate, over_detected_rate, average_IoU

def GetArgs():
    parser = argparse.ArgumentParser(description='Evaluate the performance for final prediction')
    parser.add_argument('-d', dest='abs_distance', 
        help='absolute condition for matching bbox', default=np.inf, type=int)
    parser.add_argument('-i', dest='IoU',
        help='relative condition for matching bbox', default=0.5, type=float)
    parser.add_argument('-isbox', dest='isbox',
        help='evaluation with box range or not', default=1, type=int)
    parser.add_argument('-cfg', dest='cfg_EvalFinal', 
        help='config file for dataset', default=None, type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args
        
if __name__=="__main__":
    
    tic = time.clock()
    
    global abs_distance, unIoU, categories, conf, is_box_range
   
    if '-cfg' in sys.argv:
        args = GetArgs()
        abs_distance = args.abs_distance
        unIoU = 1 - args.IoU
        is_box_range = args.isbox
       
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
        abs_distance = np.inf
        unIoU = 0.5
        is_box_range = 0

        if '-c' in sys.argv:
            categories = GetCategories(sys.argv[sys.argv.index('-c')+1])    
        if '-d' in sys.argv:
            abs_distance = float(sys.argv[sys.argv.index('-d')+1])
        if '-i' in sys.argv:
            unIoU = 1- float(sys.argv[sys.argv.index('-i')+1])
        if '-isbox' in sys.argv:
            is_box_range = int(sys.argv[sys.argv.index('-isbox')+1])

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

    df_class_infs = ClassInformationToCsv(results) 
    precision, recall, F_score, error_rate, \
       missing_rate, over_detected_rate, average_IoU = PerformanceToCsv(results) 

    print("*************************************************************************")
    print(df_class_infs) 
    print("*************************************************************************")
    print("precision:{:.3f}, recall:{:.3f}, F_score:{:.3f}, error_rate:{:.3f}"\
       .format(precision, recall, F_score, error_rate)) 
    print("missing_rate:{:.3f}, over_detected_rate:{:.3f}, average_IoU:{:.3f}"\
       .format(missing_rate, over_detected_rate, average_IoU))
    print("*************************************************************************") 
   
    toc = time.clock()
    print('running time: {:.3f} seconds'.format(toc-tic))
    
