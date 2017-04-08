# -*- coding:UTF-8 -*-
import numpy as np
from numpy import array
from multiprocessing import Pool
import os, sys, argparse, time, cv2

# python python/shoes/get_bbox_labels_by_canny.py -i data/shoes/JPEGImages -o data/shoes/labels -t data/shoes/train.txt

def get_file_list(input_root, output_root):
    input_img_list = []
    output_txt_list = []   
    if not os.path.exists(output_root):
        os.mkdir(output_root)     
    for temp_dir in os.listdir(input_root):
        input_dir = os.path.join(input_root, temp_dir)
        output_dir = os.path.join(output_root, temp_dir.replace('jpg', 'txt'))
        if input_dir.endswith('.jpg'):
            input_img_list.append(input_dir)
            output_txt_list.append(output_dir)
        elif os.path.isdir(input_dir):
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            for input_file in os.listdir(input_dir):
                if input_file.endswith('.jpg'):
                    input_img_list.append(os.path.join(input_dir, input_file))
                    output_txt_list.append(os.path.join(output_dir, input_file))
        else:
            pass
    return input_img_list, output_txt_list

def get_bbox_labels(arguments):
    input_img_file, output_txt_file = arguments[0], arguments[1]
    img = cv2.imread(input_img_file)
    grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayed = cv2.blur(grayed, (3, 3))
    width = grayed.shape[1]
    height = grayed.shape[0]
    canimg = cv2.Canny(grayed, 50, 80)

    if np.max(canimg) == 0:
        top    = 0
        right  = width - 1
        bottom = height - 1
        left   = 0
    else:
        linepix = np.where(canimg == 255)
        top    = min(linepix[0])
        right  = max(linepix[1])
        bottom = max(linepix[0])
        left   = min(linepix[1])
    width, height = float(width), float(height)
    top, right, bottom, left = top/height, right/width, bottom/height, left/width
    line = array([[0, top, right, bottom, left]])
    np.savetxt(output_txt_file, line, fmt="%d %f %f %f %f")

def get_args():
    parser = argparse.ArgumentParser(description = 'draw the bbox of images')
    parser.add_argument('-i', dest = 'input_root',
        help = 'input root of images', default = None, type = str)
    parser.add_argument('-o', dest = 'output_root', 
        help = 'output root of labels', default = None, type = str)
    parser.add_argument('-t', dest = 'train_txt', 
        help = 'train.txt', default = None, type = str)    
    parser.add_argument('-c', dest = 'cpu_num', 
        help = 'cpu number for multiprocessing', default = 16, type = int)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    args = get_args()

    input_root  = args.input_root
    output_root = args.output_root
    train_txt   = args.train_txt
    cpu_num     = args.cpu_num

    tic = time.time()

    input_img_list, output_txt_list = get_file_list(input_root, output_root)
    pool = Pool(cpu_num)
    pool.map(get_bbox_labels, zip(input_img_list, output_txt_list))

    if train_txt:
        input_img_list = [img_file + '\n' for img_file in input_img_list] 
        open(train_txt, 'w').writelines(input_img_list)

    toc = time.time()
    print 'running time: {} seconds'.format(toc-tic)