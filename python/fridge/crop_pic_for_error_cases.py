#********************
#date:2016-08-26
#author:luolongqiang
#********************

#cmd line: python crop_pic_for_error_cases.py -intxt eval/error_cases.txt  -injpg  eval/jpg  -out  eval/error_cases_0826 [-mode crop(draw)]


import numpy as np
import os, sys, argparse, time, cv2

def mkdir_for_pic(error_case_txt, output_dir):
    dir_list = []
    with open(error_case_txt,'r') as fi:
        for line in fi:
            s = line.strip().split(' ')
            dir_list.append(s[1]+'-to-'+s[2])
    dir_set = np.unique(dir_list)   
    dir_dict = {ele:str(dir_list.count(ele)) for ele in dir_set}
    for ele in dir_set:
        output_jpg_dir = os.path.join(output_dir, dir_dict[ele] + '-' + ele)
        if not os.path.exists(output_jpg_dir):
            os.mkdir(output_jpg_dir) 
    return dir_dict

def get_rigion_box(error_case_txt, dir_dict, input_jpg_dir, output_dir, mode):
    i = 1
    with open(error_case_txt,'r') as fi:
        for line in fi:
            obj = line.strip().split(' ')
            axes = obj[-1].split(',')
            x, y, z, w = int(float(axes[0])), int(float(axes[1])), int(float(axes[2])), int(float(axes[3]))
            im = cv2.imread(os.path.join(input_jpg_dir, obj[0].split('.')[0] + '.jpg'))
            if mode == 'crop':
                region = im[uy:ly, ux:lx]
                target = region
            else:
                cv2.rectangle(im, (x, y), (z, w), (0, 0, 255), thickness = 2)
                cv2.putText(im, obj[1], (x, y), font, 2, (255, 0, 0), thickness = 2, lineType = 8)
                cv2.putText(im, obj[2], (z, w), font, 2, (0, 255, 0), thickness = 2, lineType = 8)
                target = im                
            ele = obj[1]+'-to-'+obj[2]
            output_jpg = os.path.join(output_dir, dir_dict[ele]+'-'+ele, str(i) + '-' + obj[0].split('.')[0] + '.jpg')
            target.save(output_jpg)
            i = i + 1
    fi.close()

def get_args():
    parser = argparse.ArgumentParser(description='crop or draw pictures for error cases')
    parser.add_argument('-intxt', dest='input_txt', 
        help='error case txt', default=None, type=str)
    parser.add_argument('-injpg', dest='input_jpg_dir', 
        help='input jpg directory', default=None, type=str)
    parser.add_argument('-out', dest='output_dir', 
        help='output jpg directory for error cases', default=None, type=str)
    parser.add_argument('-mode', dest='mode', 
        help='crop or draw pictures for error cases, default=crop', default='crop', type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

if __name__=='__main__':
    tic = time.clock()
    
    args = get_args()
    mode = args.mode
    error_case_txt = args.input_txt
    input_jpg_dir = args.input_jpg_dir
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    # make jpg directory for error cases
    dir_dict = mkdir_for_pic(error_case_txt, output_dir)
    # save jpg for error cases
    get_rigion_box(error_case_txt, dir_dict, input_jpg_dir, output_dir, mode)
    
    toc = time.clock()
    print('running time:{:.3f}'.format(toc-tic))
        
    
   
        
        
        
        
        
        