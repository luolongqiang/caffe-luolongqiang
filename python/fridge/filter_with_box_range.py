#*****************************
#date:2016-07-26
#author:luolongqiang
#*****************************

import numpy as np
import pandas as pd
import os, sys, time, argparse
from multiprocessing import Pool, cpu_count
from EvalFinal import GetFileList, IsSubString

def filter_with_box_range(args):
    input_txt, output_txt, box_range = args[0], args[1], args[2]
    fi, fo = open(input_txt, 'r'), open(output_txt, 'w') 
    while True:
        s = fi.readline()
        if not s:
            break
        obj = s.strip().split(' ') 
        x = float(obj[1])
        y = float(obj[2])
        z = float(obj[1]) + float(obj[3])
        w = float(obj[2]) + float(obj[4])
        box_area = (z-x+1.0)*(w-y+1.0)
        xmax, ymax = max(box_range[0], x), max(box_range[1], y)
        zmin, wmin = min(box_range[2], z), min(box_range[3], w)
        w = max(zmin - xmax + 1.0, 0.0)
        h = max(wmin - ymax + 1.0, 0.0)
        inter_area = w * h
        box_IoU = 0
        if box_area != 0:
            box_IoU = inter_area*1.0/box_area
        global IoU
        if box_IoU > IoU:
            fo.write(s)
    fi.close()
    fo.close()
   
def GetArgs():
    parser = argparse.ArgumentParser(description='filter predicted labels by box range of test')
    parser.add_argument('-in', dest='input_root', 
        help='input-root of predicted labels', default=None, type=str)
    parser.add_argument('-out', dest='output_root', 
        help='output-root of predicted labels within box range', default=None, type=str)
    parser.add_argument('-csv', dest='box_range', 
        help='box range with format .csv file', default=None, type=str)
    parser.add_argument('-IoU', dest='IoU', 
        help='the IoU threshold of per prediction with box range', default=0.5, type=float)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

if __name__=='__main__':  
    tic = time.clock()

    global IoU

    args = GetArgs()  
    IoU =  args.IoU
    input_txt_root = args.input_root
    output_txt_root = args.output_root
    box_range_df = pd.read_csv(args.box_range)
    box_range_dict = {box_range_df['picture'][i]:[box_range_df['top_left'][i],\
                                                  box_range_df['top_right'][i],\
                                                  box_range_df['bottom_left'][i],\
                                                  box_range_df['bottom_right'][i]]\
                                                  for i in range(len(box_range_df))}

    input_txt_list = []
    output_txt_list = []
    box_range_list = []
    for txt_dir in os.listdir(input_txt_root):
        input_txt_dir = os.path.join(input_txt_root, txt_dir)
        temp_list = GetFileList(input_txt_dir, ['.txt'])
        input_txt_list += temp_list
        output_txt_dir = os.path.join(output_txt_root, txt_dir)
        if not os.path.exists(output_txt_dir):
            os.mkdir(output_txt_dir)
        for pth in temp_list:
            complete_path = os.path.join(output_txt_dir, pth.split('/')[-1])
            output_txt_list.append(complete_path) 
            jpg_name = pth.split('/')[-1].split('.')[0] + '.jpg'
            box_range_list.append(box_range_dict[jpg_name])  
   
    pool = Pool(int(cpu_count()*7/8)) 
    pool.map(filter_with_box_range, zip(input_txt_list, output_txt_list, box_range_list))  
    
    toc = time.clock()
    print('running time:{:.3f} seconds'.format(toc-tic))
      
