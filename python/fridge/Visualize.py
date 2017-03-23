#*****************************
#date:2016-08-03
#author:luolongqiang
#*****************************

import os, sys, cv2, copy, time, argparse
import numpy as np
from numpy import array
from math import sqrt
from pandas import DataFrame
from multiprocessing import Pool, cpu_count
from EvalFinal import GetFileList, IsSubString, GetLimitedLabels, GetCategories,\
    GetDistMatrixByArea, GetBoxIoU, GetMatchedLabels, GetMissingLabels
 
def ModifyMatchedLabels(r_matched_labels, p_matched_labels, r_missing_matched_labels, p_missing_matched_labels):    
    if len(r_missing_matched_labels) > 0:
        for i in range(len(r_missing_matched_labels)):
            r_obj = r_missing_matched_labels[i]
            p_obj = p_missing_matched_labels[i]
            if 'hard' in r_obj[0]:
                r_matched_labels.append([r_obj[0], r_obj[1], r_obj[2], r_obj[3], r_obj[4]])
                p_matched_labels.append([p_obj[0], p_obj[1], p_obj[2], p_obj[3], p_obj[4]])
    return r_matched_labels, p_matched_labels            

def VisualizeForBBox(input_jpg, labels, title, mark = False):
    im = cv2.imread(input_jpg)
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL     
    cv2.putText(im, title, (30,30), font, 2, (0, 0 ,255), thickness = 2, lineType = 8)
    for obj in labels:
        if obj[0] in categories:
            x, y, z, w = int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4])
            cv2.rectangle(im, (x, y), (z, w), (255, 0, 0), thickness = 2)
            if mark == True:
                cv2.putText(im, obj[0], (x, (y+w)/2), font, 2, (255, 0, 0), thickness = 2, lineType = 8)
    return im

def VisualizeForMatchedBBox(input_jpg, r_matched_labels, p_matched_labels, title):
    im = cv2.imread(input_jpg) 
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    cv2.putText(im, title, (30,30), font, 2, (0, 0 ,255), thickness = 2, lineType = 8)
    if len(r_matched_labels) == 0:
        return im;
    for i in range(len(r_matched_labels)):
        r_obj = r_matched_labels[i]
        p_obj = p_matched_labels[i]       
        rx, ry, rw, rz = int(r_obj[1]), int(r_obj[2]), int(r_obj[3]), int(r_obj[4])
        px, py, pw, pz = int(p_obj[1]), int(p_obj[2]), int(p_obj[3]), int(p_obj[4])
        r_color = (0, 0, 255)
        if r_obj[0] == p_obj[0]: 
            r_title = str(i+1) + 'Y'
            p_title = str(i+1) + 'Y'
            p_color = (0, 255, 0)
        elif 'hard' not in r_obj[0]: 
            r_title = str(i+1) + r_obj[0]  
            p_title = str(i+1) + p_obj[0]
            p_color = (255, 0, 0)  
        cv2.rectangle(im, (rx, ry), (rw, rz), r_color, thickness = 2)
        cv2.rectangle(im, (px, py), (pw, pz), p_color, thickness = 2)
        cv2.putText(im, r_title, ((rx+rw)/2, (ry+rz)/2), font, 2, r_color, thickness = 2, lineType = 8)
        cv2.putText(im, p_title, ((px+pw)/2, (py+pz)/2), font, 2, p_color, thickness = 2, lineType = 8)
    return im

def Experiment(arguments): 
    real_txt = arguments[0]
    predicted_txt = arguments[1]
    real_jpg = arguments[2]
    output_jpg = arguments[3]
                  
    r_labels = GetLimitedLabels(real_txt, categories, level = 2, justword1 = True, isall = False)
    p_labels = GetLimitedLabels(predicted_txt, categories, level = 2, justword1 = True, isall = False)  
      
    dist_matrix = GetDistMatrixByArea(r_labels, p_labels, abs_distance)
    r_matched_labels, p_matched_labels = \
        GetMatchedLabels(copy.copy(r_labels), copy.copy(p_labels), dist_matrix, [], [], unIoU)    
  
    r_all_labels = GetLimitedLabels(real_txt, categories, level = 0, justword1 = True, isall = True)
    p_all_labels = GetLimitedLabels(predicted_txt, categories, level = 0, justword1 = True, isall = True)   
    r_missing_labels = GetMissingLabels(copy.copy(r_all_labels), copy.copy(r_matched_labels))   
    p_missing_labels = GetMissingLabels(copy.copy(p_all_labels), copy.copy(p_matched_labels))  
    
    missing_dist_matrix = \
        GetDistMatrixByArea(copy.copy(r_missing_labels), copy.copy(p_missing_labels), abs_distance)
    r_missing_matched_labels, p_missing_matched_labels = \
        GetMatchedLabels(copy.copy(r_missing_labels), copy.copy(p_missing_labels), missing_dist_matrix, [], [], unIoU)    
            
    r_matched_labels, p_matched_labels = \
        ModifyMatchedLabels(r_matched_labels, p_matched_labels, r_missing_matched_labels, p_missing_matched_labels)    

    r_missing_labels = GetMissingLabels(copy.copy(r_all_labels), copy.copy(r_matched_labels))   
    p_missing_labels = GetMissingLabels(copy.copy(p_all_labels), copy.copy(p_matched_labels))
    
    im1 = VisualizeForBBox(real_jpg, r_missing_labels, 'missing bboxes', mark = False)
    im2 = VisualizeForMatchedBBox(real_jpg, r_matched_labels, p_matched_labels, 'matched bboxes')           
    im3 = VisualizeForBBox(real_jpg, p_missing_labels, 'over detected bboxes', mark = True)  
    merged_img = np.hstack((im1, im2, im3))
    cv2.imwrite(output_jpg, merged_img)            

def GetArgs():
    parser = argparse.ArgumentParser(description='Visualize the performance for final prediction')
    parser.add_argument('-d', dest='abs_distance', \
        help='absolute condition for matching bbox', default=np.inf, type=int)
    parser.add_argument('-i', dest='IoU', 
        help='relative condition for matching bbox', default=0.5, type=float)
    parser.add_argument('-cfg', dest='cfg_Visualize', 
        help='config file for dataset', default=None, type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args
        
if __name__=="__main__":
    
    tic = time.clock()
    
    global abs_distance, unIoU, categories
    
    if '-cfg' in sys.argv:
        args = GetArgs()
        abs_distance = args.abs_distance
        unIoU = 1 - args.IoU
       
        conf = {}
        with open(args.cfg_Visualize, 'r') as f:
            for line in f:
                infos = line.strip().split('=')
                conf[infos[0]] = infos[1]

        categories = conf['cls_names'].split(',')    

        real_txt_list = []
        real_jpg_list = []
        output_jpg_list = []
        real_root = conf['real_root']
        output_jpg_root = conf['output_root']
        for data_dir in conf['real_dir'].split(','):
            txt_pth = os.path.join(real_root, data_dir, 'txt')
            jpg_pth = os.path.join(real_root, data_dir, 'jpg')
            real_txt_list += GetFileList(txt_pth, ['.txt'])
            temp_jpg_list = GetFileList(jpg_pth, ['.jpg'])
            real_jpg_list += temp_jpg_list
            output_jpg_dir = output_jpg_root
            for idir in data_dir.split('/'):
                output_jpg_dir = os.path.join(output_jpg_dir, idir)
                if not os.path.exists(output_jpg_dir):
                    os.mkdir(output_jpg_dir)
            for pth in temp_jpg_list:
                complete_path = os.path.join(output_jpg_dir, 'merged_' + pth.split('/')[-1])
                output_jpg_list.append(complete_path)
        
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

        if '-c' in sys.argv:
            categories = GetCategories(sys.argv[sys.argv.index('-c')+1])    
        if '-d' in sys.argv:
            abs_distance = float(sys.argv[sys.argv.index('-t1')+1])
        if '-i' in sys.argv:
            unIoU = 1- float(sys.argv[sys.argv.index('-t2')+1])       

        real_txt_list = []
        real_jpg_list = []
        predicted_txt_list = []
        output_jpg_list = []

        real_root = sys.argv[1]
        predicted_root = sys.argv[2]
        output_jpg_root = sys.argv[3]

        for data_dir in os.listdir(sys.argv[1]):
            txt_pth = os.path.join(real_root, data_dir, 'txt')
            jpg_pth = os.path.join(real_root, data_dir, 'jpg')
            real_txt_list += GetFileList(txt_pth, ['.txt'])
            temp_jpg_list = GetFileList(jpg_pth, ['.jpg'])
            real_jpg_list += temp_jpg_list
            output_jpg_dir = output_jpg_root
            for idir in data_dir.split('/'):
                output_jpg_dir = os.path.join(output_jpg_dir, idir)
                if not os.path.exists(output_jpg_dir):
                    os.mkdir(output_jpg_dir)
            for pth in temp_jpg_list:
                complete_path = os.path.join(output_jpg_dir, 'merged_' + pth.split('/')[-1])
                output_jpg_list.append(complete_path)

        for data_dir in os.listdir(sys.argv[2]):
            complete_pth = os.path.join(predicted_root, data_dir)
            predicted_txt_list += GetFileList(complete_pth, ['.txt'])
     
    pool = Pool(int(cpu_count()*7/8))
    pool.map(Experiment, zip(real_txt_list, predicted_txt_list, real_jpg_list, output_jpg_list))
    
    toc = time.clock()
    print('running time: {:.3f} seconds'.format(toc-tic))
    
