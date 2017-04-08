# -*- coding:UTF-8 -*-
import numpy as np
from numpy import array 
from PIL import Image
from multiprocessing import Pool
import os, sys, argparse, time, cv2, shutil, random

# python python/shoes/get_data_by_edge.py -i data/shoes/img3 -b data/shoes/background -o data/shoes

img_dir = 'JPEGImages'
lbs_dir = 'labels'

def get_file_list(input_dir, backgr_dir, output_root):
    if not os.path.exists(output_root):
        os.mkdir(output_root) 

    backgr_img_set = []
    for temp_img in os.listdir(backgr_dir):
        if temp_img.endswith('.jpg'):
            backgr_img = os.path.join(backgr_dir, temp_img)
            backgr_img_set.append(backgr_img)    

    input_img_list = []
    output_img_list = []
    output_txt_list = [] 
    for temp_img in os.listdir(input_dir):
        if temp_img.endswith('.jpg'):
            input_img = os.path.join(input_dir, temp_img)
            temp_name = temp_img[:temp_img.rfind('.')]
            output_img = os.path.join(output_root, img_dir, temp_name+'-add.jpg')
            output_txt = os.path.join(output_root, lbs_dir, temp_name+'-add.txt')
            input_img_list.append(input_img)
            output_img_list.append(output_img)
            output_txt_list.append(output_txt)
    
    output_num = len(output_img_list)
    backgr_num = len(backgr_img_set)
    times = output_num/backgr_num + 1
    backgr_img_list = random.sample(times*backgr_img_set, output_num)

    return input_img_list, backgr_img_list, output_img_list, output_txt_list

def get_img_edge((input_img_file, backgr_img_file, output_img_file, output_txt_file)):
    print input_img_file
    img = cv2.imread(input_img_file)
    top, right, bottom, left, canimg = get_bbox_by_canny(img)
    
    box_w,  box_h = right-left, bottom-top
    if box_w*box_h > img.shape[1]*img.shape[0]/16.0:
        re_img = np.ones((4*box_h, 4*box_w, img.shape[2]), np.uint8)*255
        re_top, re_right, re_bottom, re_left = \
           box_h*3/2, box_w*5/2, box_h*5/2, box_w*3/2 
        re_img[re_top:re_bottom, re_left:re_right, :] = img[top:bottom, left:right, :]
        top, right, bottom, left, canimg = get_bbox_by_canny(re_img)
        img = re_img

    width, height = img.shape[1], img.shape[0]
    output_labels(top, right, bottom, left, width, height, output_txt_file)

    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (left, top, right - left, bottom - top) 

    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
    img = img*mask2[:, :, np.newaxis]
     
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (199, 199))
    th3 = cv2.dilate(canimg, kernel)
    bin_img = cv2.bitwise_and(th3, th3, mask=mask2)
    
    cv2.imwrite('temp.png', img)
    cv2.imwrite('temp.bmp', bin_img) 
    img = Image.open('temp.png')
    bin_img = Image.open('temp.bmp')
    bg_img = Image.open(backgr_img_file)
    
    img = img.convert("RGBA")
    bg_img = bg_img.convert("RGBA")
    bg_img = bg_img.resize(img.size)
    bg_img.paste(img, (0, 0, img.size[0], img.size[1]), bin_img)
    bg_img.save(output_img_file)

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
    return top, right, bottom, left, canimg

def output_labels(top, right, bottom, left, width, height, output_txt_file):
    x = (left + right)/2.0
    y = (top + bottom)/2.0
    w = right - left
    h = bottom - top
    x, y, w, h = x/width, y/height, w*1.0/width, h*1.0/height
    line = array([[0, x, y, w, h]])
    np.savetxt(output_txt_file, line, fmt="%d %f %f %f %f")

def get_args():
    parser = argparse.ArgumentParser(description = 'get shoes data')
    parser.add_argument('-i', dest = 'input_dir',
        help = 'input dir of images', default = None, type = str)
    parser.add_argument('-b', dest = 'backgr_dir',
        help = 'background dir of images', default = None, type = str)  
    parser.add_argument('-o', dest = 'output_root', 
        help = 'output root of JPEGImages and labels', default = None, type = str)  
    parser.add_argument('-c', dest = 'cpu_num', 
        help = 'cpu number', default = 8, type = int) 
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    args = get_args()

    input_dir   = args.input_dir
    backgr_dir  = args.backgr_dir
    output_root = args.output_root
    cpu_num     = args.cpu_num

    tic = time.time()

    input_img_list, backgr_img_list, output_img_list, output_txt_list = \
        get_file_list(input_dir, backgr_dir, output_root)
    '''
    pool = Pool(cpu_num)
    pool.map(get_img_edge, zip(input_img_list, backgr_img_list, \
       output_img_list, output_txt_list))
    '''
    for arguments in zip(input_img_list, backgr_img_list, output_img_list, output_txt_list):
        get_img_edge(arguments)
    
    toc = time.time()
    print 'running time: {} seconds'.format(toc-tic)