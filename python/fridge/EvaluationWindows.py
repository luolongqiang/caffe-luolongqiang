#*********************
#data:2016-08-04
#author:luolongqiang
#*********************

import numpy as np
from numpy import array
from math import sqrt
from PIL import Image,ImageDraw,ImageFont
from pandas import DataFrame
import time, sys, os
 
def GetFileList(FindPath,FlagStr=[]):     
    FileList=[]  
    FileNames=os.listdir(FindPath)  
    if (len(FileNames)>0):  
       for fn in FileNames:  
           if (len(FlagStr)>0):  
               if (IsSubString(FlagStr,fn)):  
                   fullfilename=os.path.join(FindPath,fn)  
                   FileList.append(fullfilename)  
           else:  
               fullfilename=os.path.join(FindPath,fn)  
               FileList.append(fullfilename)  
    if (len(FileList)>0):  
        FileList.sort()    
    return FileList 
    
def IsSubString(SubStrList,Str):  
    flag=True  
    for substr in SubStrList:  
        if not(substr in Str):  
            flag=False  
    return flag

def GetLimitedLabels(input_txt, level = 2, justword1 = True, isall = False):
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
                    if '-hard' or '-l2' not in obj[0]:
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
    
def GetPerClassInformation(r_labels, p_labels, class_infs):
    for ele in r_labels:
        class_infs[ele[0]][0] += 1
    for ele in p_labels:
        class_infs[ele[0]][1] += 1
    return class_infs  
    
def GetDistMatrixByDiagLine(r_labels, p_labels):
    n, m = len(r_labels), len(p_labels) 

    diag_lens1 = [sqrt((r_labels[i][1]-r_labels[i][3])**2+(r_labels[i][2]-r_labels[i][4])**2) for i in range(n)]  
    diag_lens2 = [sqrt((p_labels[i][1]-p_labels[i][3])**2+(p_labels[i][2]-p_labels[i][4])**2) for i in range(m)]
    
    dist_matrix = np.ones((n,m))
    for i in range(n):
        x1, y1 = r_labels[i][1], r_labels[i][2]
        z1, w1 = r_labels[i][3], r_labels[i][4] 
        for j in range(m):        
            d1 = sqrt((x1-p_labels[j][1])**2 + (y1-p_labels[j][2])**2)
            d2 = sqrt((z1-p_labels[j][3])**2 + (w1-p_labels[j][4])**2)
            if d1 < abs_distance and d2 < abs_distance:
                dist_matrix[i,j] = (d1+d2)/(diag_lens1[i]+diag_lens2[j])
    return dist_matrix

def GetDistMatrixByArea(r_labels, p_labels):
    n, m = len(r_labels), len(p_labels)
    dist_matrix = np.ones((n, m))
    for i in range(n):
        x1, y1 = r_labels[i][1], r_labels[i][2]
        z1, w1 = r_labels[i][3], r_labels[i][4] 
        box1_area = (z1-x1+1.0)*(w1-y1+1.0)
        for j in range(m):  
            x2, y2 = p_labels[j][1], p_labels[j][2]
            z2, w2 = p_labels[j][3], p_labels[j][4]
            d1 = sqrt((x1-x2)**2 + (y1-y2)**2)
            d2 = sqrt((z1-z2)**2 + (w1-w2)**2)
            if d1 < abs_distance and d2 < abs_distance:
                box2_area = (z2-x2+1.0)*(w2-y2+1.0)
                ixmin, iymin = max(x1,x2), max(y1,y2)
                ixmax, iymax = min(z1,z2), min(w1,w2)
                iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
                ih = np.maximum(iymax - iymin + 1.0, 0.0)
                inter_area = iw * ih
                union_area = box1_area + box2_area - inter_area
                if union_area != 0:
                    dist_matrix[i,j] = 1 - inter_area/union_area
    return dist_matrix    

def GetMatchedLabels(r_labels, p_labels, dist_matrix, r_matched_labels, p_matched_labels):
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
        return GetMatchedLabels(r_labels, p_labels, dist_matrix, r_matched_labels, p_matched_labels)        

def GetMissingLabels(all_labels, matched_labels):
    if len(matched_labels)>0:
        for ele in matched_labels:
            all_labels.remove(ele)
    return all_labels

def EvaluatePerformance(n, m, r_matched_labels, p_matched_labels, r_missing_matched_labels, p_missing_matched_labels, class_infs): 
    global N, M, K, L, J
    N, M = N+n, M+m
    if len(r_matched_labels) == 0:
        return m, 0., 0., 0., 0., 0., 1., 1., 1., r_matched_labels, p_matched_labels, class_infs
    r_names = array(r_matched_labels)[:,0]
    p_names = array(p_matched_labels)[:,0]
    k = len(r_names)
    l = sum(r_names == p_names)  
    for i in range(k):
        class_infs[r_names[i]][2] += 1
        class_infs[p_names[i]][3] += 1
        if r_names[i]==p_names[i]:
            class_infs[r_names[i]][4] += 1   
    j = 0 
    if len(r_missing_matched_labels) > 0:
        for i in range(len(r_missing_matched_labels)):
            r_obj = r_missing_matched_labels[i]
            p_obj = p_missing_matched_labels[i]
            if 'hard' in r_obj[0]:
                j = j + 1
                r_matched_labels.append([r_obj[0], r_obj[1], r_obj[2], r_obj[3], r_obj[4]])
                p_matched_labels.append([p_obj[0], p_obj[1], p_obj[2], p_obj[3], p_obj[4]])
                class_infs[p_obj[0]][1] -= 1
    K, L, J, m = K+k, L+l, J+j, m-j
    precision, recall = l*1./m, l*1./n
    F_score = 2*precision*recall*1.0/(precision + recall)
    error_rate = 1-l*1./k
    missing_rate = (n-k)*1./n
    over_detected_rate = (m-k)*1./m
    return m, k, l, precision, recall, F_score, error_rate, missing_rate, over_detected_rate, r_matched_labels, p_matched_labels, class_infs             
    
def VisualizeForBBox(image_file_path, labels, color, title, ft, mark=False):
    im = Image.open(image_file_path)
    draw = ImageDraw.Draw(im)
    draw.text([10,0], title, font=ft, fill=color)
    for ele in labels:
        if ele[0] in categories:
            ux, uy, lx, ly = ele[1], ele[2], ele[3], ele[4]
            draw.line(((ux, uy), (lx, uy), (lx, ly), (ux, ly), (ux, uy)), fill=color, width=3)
            if mark == True:
                draw.text([ux,(uy+ly)*1./2], ele[0], font=ft, fill=color)
    return im

def VisualizeForMatchedBBox(image_file_path, r_matched_labels, p_matched_labels, title, ft):
    im = Image.open(image_file_path) 
    draw = ImageDraw.Draw(im)
    draw.text([10, 0], title, font = ft, fill = 'blue')
    if len(r_matched_labels) == 0:
        return im;
    for i in range(len(r_matched_labels)):
        r_obj = r_matched_labels[i]
        p_obj = p_matched_labels[i]
        rx, ry, rw, rz = r_obj[1], r_obj[2], r_obj[3], r_obj[4]
        px, py, pw, pz = p_obj[1], p_obj[2], p_obj[3], p_obj[4]   
        r_color = 'red'   
        if r_obj[0] == p_obj[0]:
            r_title = str(i+1) + 'Y'
            p_title = str(i+1) + 'Y'
            p_color = 'green'
        elif r_obj[0] in categories:
            r_title = str(i+1) + 'N-' + r_obj[0]
            p_title = str(i+1) + 'N-' + p_obj[0]
            p_color = 'blue'
        draw.line(((rx, ry), (rw, ry), (rw, rz), (rx, rz), (rx, ry)), fill = r_color, width=3) 
        draw.line(((px, py), (pw, py), (pw, pz), (px, pz), (px, py)), fill = p_color, width=3)   
        draw.text([(rx + rw)*1.0/2, (rz + ry)*1.0/2], r_title, font = ft, fill = r_color)
        draw.text([px, py], p_title, font = ft, fill = p_color)            
    return im

def MergeImage(imglist, outputfile):
    num = len(imglist)
    w, h = imglist[0].size
    merge_img = Image.new('RGB', (w*num, h), 0xffffff)
    i=0
    for img in imglist:
        merge_img.paste(img, (i, 0))
        i += w
    merge_img.save(outputfile)         
        
def Experiment(input_file, output_file, categories, need_jpg, level):
    real_txt_list = GetFileList(input_file + '\\txt',['.txt'])
    predicted_txt_list = GetFileList(input_file + '\\predicted_txt',['.txt'])
    
    if need_jpg == 1:    
        jpg_list = GetFileList(input_file + '\\jpg', FlagStr=['.jpg'])
        output_image_file = output_file+'\\merged_jpg'
        if not os.path.exists(output_image_file):
            os.mkdir(output_image_file)    
    
    results = []
    class_infs = {ele:list(-np.zeros(11)) for ele in categories}    
    for i in range(len(real_txt_list)):  
        real_txt = real_txt_list[i]
        predicted_txt = predicted_txt_list[i]
                  
        r_labels = GetLimitedLabels(real_txt, level = level, justword1 = True, isall = False)
        p_labels = GetLimitedLabels(predicted_txt, level = level, justword1 = True, isall = False) 
        class_infs = GetPerClassInformation(r_labels, p_labels, class_infs)
        
        n, m = len(r_labels), len(p_labels)
        dist_matrix = GetDistMatrixByArea(r_labels, p_labels)
        r_matched_labels, p_matched_labels = GetMatchedLabels(r_labels.copy(), p_labels.copy(), dist_matrix, [], [])    
        
        r_all_labels = GetLimitedLabels(real_txt, level = 0, justword1 = True, isall = True)
        p_all_labels = GetLimitedLabels(predicted_txt, level = 0, justword1 = True, isall = True)  

        r_missing_labels = GetMissingLabels(r_all_labels.copy(), r_matched_labels.copy())   
        p_missing_labels = GetMissingLabels(p_all_labels.copy(), p_matched_labels.copy())  
        missing_dist_matrix = GetDistMatrixByArea(r_missing_labels.copy(), p_missing_labels.copy())
        r_missing_matched_labels, p_missing_matched_labels = \
           GetMatchedLabels(r_missing_labels.copy(), p_missing_labels.copy(), missing_dist_matrix, [], [])    
                
        m, k, l, precision, recall, F_score, error_rate, missing_rate, over_detected_rate, r_matched_labels, p_matched_labels, class_infs = \
           EvaluatePerformance(n, m, r_matched_labels, p_matched_labels, r_missing_matched_labels, p_missing_matched_labels, class_infs)       
        results.append([real_txt.split('\\')[-1].split('.')[0], n, m, k, l, precision, recall, F_score, error_rate, missing_rate, over_detected_rate ])
        
        if need_jpg == 1:    
            jpg = jpg_list[i] 
            ft1 = ImageFont.truetype("C:\\Windows\\Fonts\\simsunb.ttf",36)
            r_missing_labels = GetMissingLabels(r_all_labels, r_matched_labels)   
            p_missing_labels = GetMissingLabels(p_all_labels, p_matched_labels)
            im1 = VisualizeForBBox(jpg, r_missing_labels, 'red', 'missing bounding box', ft1, mark = False)
            im2 = VisualizeForBBox(jpg, p_missing_labels, 'green','over detected bounding box', ft1, mark = True)
            im3 = VisualizeForMatchedBBox(jpg, r_matched_labels, p_matched_labels, 'matched bounding box', ft1)           
            MergeImage([im1, im3, im2], output_image_file+'\\merged_' + jpg.split('\\')[-1])     
    return results, class_infs
        
if __name__=="__main__":
    
    tic = time.clock()
    
    global categories, N, M, K, L, J, abs_distance, unIoU, input_file, output_file
    categories = ['apple','beer','broccoli','chineseCabbage','cucumber','egg',\
                  'eggplant','grape','ham','milk','onion','orange','bitter','pear',\
                  'potato','radish','strawberry','tomato','watermelon','whiteradish']
    N, M, K, L, J = 0., 0., 0., 0., 0.

    abs_distance = np.inf
    unIoU = 0.5
    need_jpg = 0
    level = 2

    if '-c' in sys.argv:
        categories = GetCategories(sys.argv[sys.argv.index('-c')+1])
    if '-t1' in sys.argv:
        diag_dist = float(sys.argv[sys.argv.index('-t1')+1])
    if '-t2' in sys.argv:
        threshold = 1- float(sys.argv[sys.argv.index('-t2')+1])
    if '-p' in sys.argv:
        need_jpg = int(sys.argv[sys.argv.index('-p')+1])
    if '-l' in sys.argv:
        level = int(sys.argv[sys.argv.index('-l')+1])
      
    input_file, output_file = sys.argv[1], sys.argv[2]
    results, class_infs = Experiment(input_file, output_file, categories, need_jpg, level)

    M = M - J
    precision, recall, F_score =0., 0., 0.
    error_rate, missing_rate, over_detected_rate = 1., 1., 1.
    if M != 0:
        precision = L/M
        over_detected_rate = (M-L)/M
    if N != 0:
        recall = L/N
        missing_rate = (N-K)/N
    if K != 0:
        error_rate = 1-L/K
    if precision + recall != 0:
        F_score = 2*precision*recall/(precision+recall)
    print("***********************************************************************")
    print("precision:{:.3f}; recall:{:.3f}; F_score:{:.3f}; error_rate:{:.3f}; missing_rate:{:.3f}; over_detected_rate:{:.3f}"\
          .format(precision, recall, F_score, error_rate, missing_rate, over_detected_rate))
    print("***********************************************************************") 
    
    results.append(['total', N, M, K, L, precision, recall, F_score, error_rate, missing_rate, over_detected_rate])    
    results = array(results)
    results = DataFrame({'picture':results[:,0],\
                       'real_number':results[:,1],\
                       'predicted_number':results[:,2],\
                       'matched_box_number':results[:,3],\
                       'matched_label_number':results[:,4],\
                       'precision':results[:,5],\
                       'recall':results[:,6],\
                       'F_score':results[:,7],\
                       'error_rate':results[:,8],\
                       'missing_rate':results[:,9],\
                       'over_detected_rate':results[:,10]})
    results = results[['picture','real_number','predicted_number','matched_box_number','matched_label_number',\
                           'precision','recall','F_score','error_rate','missing_rate','over_detected_rate']]    
    results.to_csv(output_file+'\\ResultsofEvaluation.csv',index=False)    
       
    for ele in categories:
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
    
    df = {'categories':categories}
    df_header = ['real_number','predicted_number','matched_real_box_number','matched_predicted_box_number',\
       'matched_label_number','precision','recall','F_score','error_rate','missing_rate','over_detected_rate']   
    for i in range(len(df_header)):
        df[df_header[i]] = [class_infs[ele][i] for ele in categories]
    df = DataFrame(df)
    df = df[['categories']+df_header]
    df.to_csv(output_file+'\\resultclass20.csv', index=False)
    
    toc = time.clock()
    print('running time: {:.3f} seconds'.format(toc-tic))
    