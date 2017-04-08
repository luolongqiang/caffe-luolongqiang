# -*- coding:UTF-8 -*-
import numpy as np
from numpy import array
import os, sys, argparse, time, cv2, shutil, random

# python python/shoes/get_data_by_canny.py -i data/shoes/img -o data/shoes

def construct_dataset(input_dir, output_root):
    if not os.path.exists(output_root+'/JPEGImages'):
        os.makedirs(output_root+'/JPEGImages')
    if not os.path.exists(output_root+'/labels'):
        os.makedirs(output_root+'/labels')

    output_txt_list1, output_txt_list2 = [], []
    for temp_img_file in os.listdir(input_dir):
        input_img_file = os.path.join(input_dir, temp_img_file)
        output_img_file1 = os.path.join(output_root, 'JPEGImages', temp_img_file)
        output_img_file2 = output_img_file1.split('.')[0] + '-add.jpg'
        output_txt_list1.append(output_img_file1+'\n')
        output_txt_list2.append(output_img_file2+'\n')

        shutil.copy(input_img_file, output_img_file1)

        img = cv2.imread(input_img_file)
        top, right, bottom, left = get_bbox_by_canny(img)
        output_img2 = np.ones(img.shape)*random.randint(160, 250)
        output_img2[top:bottom, left:right, :] = img[top:bottom, left:right, :]
        cv2.imwrite(output_img_file2, output_img2)

        output_txt_file1 = os.path.join(output_root, 'labels', temp_img_file.replace('jpg', 'txt'))
        output_txt_file2 = output_txt_file1.split('.')[0] + '-add.txt'

        width, height = float(img.shape[1]), float(img.shape[0])
        x = (left + right)/2.0
        y = (top + bottom)/2.0
        w = right - left
        h = bottom - top
        x, y, w, h = x/width, y/height, w*1.0/width, h*1.0/height
        line = array([[0, x, y, w, h]])
        np.savetxt(output_txt_file1, line, fmt="%d %f %f %f %f")
        np.savetxt(output_txt_file2, line, fmt="%d %f %f %f %f")
    '''
    test_lines1 = random.sample(output_txt_list1, int(len(output_txt_list1)*0.06))
    test_lines2 = random.sample(output_txt_list2, int(len(output_txt_list2)*0.06))
    train_lines1 = list(set(output_txt_list1) - set(test_lines1))
    train_lines2 = list(set(output_txt_list2) - set(test_lines2))
    open(output_root+'/train.txt', 'w').writelines(train_lines1+train_lines2)
    open(output_root+'/test.txt', 'w').writelines(test_lines1+test_lines2)
    '''

def get_bbox_by_canny(img):
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
    return top, right, bottom, left

def get_args():
    parser = argparse.ArgumentParser(description = 'draw the bbox of images')
    parser.add_argument('-i', dest = 'input_dir',
        help = 'input dir of images', default = None, type = str)
    parser.add_argument('-o', dest = 'output_root', 
        help = 'output root of JPEGImages, labels', default = None, type = str)  
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    args = get_args()

    input_dir   = args.input_dir
    output_root = args.output_root

    tic = time.time()
    construct_dataset(input_dir, output_root)
    toc = time.time()

    print 'running time: {} seconds'.format(toc-tic)