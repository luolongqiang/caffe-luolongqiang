# -*- coding:UTF-8 -*-
import numpy as np
from numpy import array 
from PIL import Image
from multiprocessing import Pool
import os, sys, argparse, time, cv2, shutil, random

# python python/shoes/get_img_edge.py -i data/shoes/img10 -b data/shoes/background -o data/shoes

img_dir = 'JPEGImages'
lbs_dir = 'labels'
temp_img_file1 = 'temp.png'
temp_img_file2 = 'temp.bmp'
canny_thres1 = 30 # 254 
canny_thres2 = 90 # 255
re_size = 800
kernel_size = 199

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

def get_img_edge(input_img_file, backgr_img_file, output_img_file, output_txt_file):
    print input_img_file
    img = cv2.imread(input_img_file)
    img = fit_size(img, re_size)
    canny_img = get_canny_img(img)
    top, right, bottom, left = get_bbox(canny_img)
    
    box_w, box_h = right-left, bottom-top
    if box_w*box_h > img.shape[1]*img.shape[0]/9.0:
        img, top, right, bottom, left = get_resize_img(img, top, right, bottom, left)
        canny_img = get_canny_img(img)
 
    width, height = img.shape[1], img.shape[0]

    # output bbox labels
    output_labels(top, right, bottom, left, width, height, output_txt_file)
    
    # get mask image
    mask = np.zeros((height, width), np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (left, top, right - left, bottom - top) 
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==0)|(mask==2), 0, 1).astype('uint8')
    mask_img = img*mask2[:, :, np.newaxis]
    # temp_img_file1 = output_img_file.replace('jpg', 'png')
    cv2.imwrite(temp_img_file1, mask_img)
    mask_img = Image.open(temp_img_file1)
    mask_img = mask_img.convert("RGBA")
    
    # get binary image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size)) # 数值越大， bbox越满
    th3 = cv2.dilate(canny_img, kernel)
    bin_img = cv2.bitwise_and(th3, th3, mask=mask2)
    # temp_img_file2 = output_img_file.replace('jpg', 'bmp')
    cv2.imwrite(temp_img_file2, bin_img) 
    bin_img = Image.open(temp_img_file2)

    # get final image by mask_image + background_img
    bg_img = Image.open(backgr_img_file)
    bg_img = bg_img.convert("RGBA")
    bg_img = bg_img.resize(mask_img.size)
    bg_img.paste(mask_img, (0, 0, mask_img.size[0], mask_img.size[1]), bin_img)
    bg_img.save(output_img_file)

def fit_size(img, size):
    img_w = img.shape[1]
    img_h = img.shape[0]
    
    if img_w >= img_h:
        scale = float(size)/float(img_h)
        fit_w = max(int(scale*img_w),1)   
        fit_h = size
    else:
        scale = float(size)/float(img_w)
        fit_w = size
        fit_h = max(int(scale*img_h),1)  
    
    resized_img = cv2.resize(img, (fit_w,fit_h), interpolation = cv2.INTER_LINEAR)
    return resized_img

def get_canny_img(img):
    grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayed = cv2.blur(grayed, (3, 3))
    width = grayed.shape[1]
    height = grayed.shape[0]
    canny_img = cv2.Canny(grayed, canny_thres1, canny_thres2) # 数值越小，bbox越大
    return canny_img

def get_bbox(canny_img):
    if np.max(canny_img) == 0:
        top    = 0
        right  = width - 1
        bottom = height - 1
        left   = 0
    else:
        linepix = np.where(canny_img == 255)
        top    = min(linepix[0])
        right  = max(linepix[1])
        bottom = max(linepix[0])
        left   = min(linepix[1])
    return top, right, bottom, left

def get_resize_img(img, top, right, bottom, left):
    box_w, box_h = right-left, bottom-top
    if box_w >= box_h:
        re_img = np.ones((4*box_w, 4*box_w, 3), np.uint8)*255
        re_left = box_w*3/2
        re_top = box_w*2 - box_h/2
    else:
        re_img = np.ones((4*box_h, 4*box_h, 3), np.uint8)*255
        re_left = box_h*2 - box_w/2
        re_top = box_h*3/2
    re_right = re_left + box_w
    re_bottom = re_top + box_h
        
    re_img[re_top:re_bottom, re_left:re_right, :] = img[top:bottom, left:right, :]
    return re_img, re_top, re_right, re_bottom, re_left

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

    tic = time.time()

    input_img_list, backgr_img_list, output_img_list, output_txt_list = \
        get_file_list(input_dir, backgr_dir, output_root)
    for input_img_file, backgr_img_file, output_img_file, output_txt_file in \
           zip(input_img_list, backgr_img_list, output_img_list, output_txt_list):
        get_img_edge(input_img_file, backgr_img_file, output_img_file, output_txt_file)

    toc = time.time()
    print 'running time: {} seconds'.format(toc-tic)