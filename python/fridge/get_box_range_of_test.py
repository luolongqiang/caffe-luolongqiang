#*****************************
#date:2016-07-22
#author:luolongqiang
#*****************************

import numpy as np
from numpy import array
from pandas import DataFrame
import os, sys, cv2, time, argparse
from multiprocessing import Pool, cpu_count
from EvalFinal import IsSubString, GetFileList

def get_axes_of_box_range(input_txt):
    fi = open(input_txt, 'r')
    labels = []
    while True:
        s = fi.readline()
        if not s:
            break
        obj=s.strip().split(' ')
        x = int(obj[1])
        y = int(obj[2])
        z = int(obj[1]) + int(obj[3])
        w = int(obj[2]) + int(obj[4])
        labels.append([x,y,z,w])
    fi.close()
    labels = array(labels)  
    return min(labels[:,0]), min(labels[:,1]), max(labels[:,2]), max(labels[:,3])

def get_box_range(args):
    input_jpg, input_txt = args[0], args[1]
    x, y, z, w = get_axes_of_box_range(input_txt)
    if is_draw:
        output_jpg = args[2]
        im = cv2.imread(input_jpg)    
        cv2.rectangle(im, (x, y), (z, w), (0, 0, 255), thickness = 3)
        cv2.imwrite(output_jpg, im)   
    return [input_jpg.split('/')[-1], x, y, z, w]
   
def GetArgs():
    parser = argparse.ArgumentParser(description='get box range of test')
    parser.add_argument('-in', dest='input_root', 
        help='input-root of real labels and pictures', default=None, type=str)
    parser.add_argument('-out', dest='output_root', 
        help='output-root of box range of per picture', default=None, type=str)
    parser.add_argument('-isDraw', dest='is_draw', 
        help='draw box range or not', default=0, type=int)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

if __name__=='__main__':
    tic = time.clock()  
    global is_draw, output_txt

    args = GetArgs()
    input_root = args.input_root
    output_root = args.output_root   
    is_draw = args.is_draw

    pool = Pool(int(cpu_count()*7/8))
     
    for input_dir in os.listdir(input_root):
        input_txt_list = []
        input_jpg_list = []
        output_jpg_list = []
        input_txt_dir = os.path.join(input_root, input_dir, 'txt')
        input_jpg_dir = os.path.join(input_root, input_dir, 'jpg') 
        input_txt_list += GetFileList(input_txt_dir, ['.txt'])         
        temp_list =  GetFileList(input_jpg_dir, ['.jpg'])
        input_jpg_list += temp_list
        output_dir = os.path.join(output_root, input_dir)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        if is_draw:
            output_jpg_dir = os.path.join(output_dir, 'jpg')
            if not os.path.exists(output_jpg_dir):
                os.mkdir(output_jpg_dir)        
            for pth in temp_list:
                complete_path = os.path.join(output_jpg_dir, pth.split('/')[-1])
                output_jpg_list.append(complete_path)
            results = pool.map(get_box_range, zip(input_jpg_list, input_txt_list, output_jpg_list))
        else:
            results = pool.map(get_box_range, zip(input_jpg_list, input_txt_list))       
        results = array(results)
        df = DataFrame({'picture':results[:,0],\
                        'top_left':results[:,1],\
                        'top_right':results[:,2],\
                        'bottom_left':results[:,3],\
                        'bottom_right':results[:,4]})
        df = df[['picture','top_left','top_right','bottom_left','bottom_right']]
        df.to_csv(output_dir + '/box_range.csv', index = False)
        
    toc = time.clock()
    print('running time:{:.3f} seconds'.format(toc-tic))



