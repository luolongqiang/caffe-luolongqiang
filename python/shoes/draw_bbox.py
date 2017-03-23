# -*- coding:UTF-8 -*-
import numpy as np
from PIL import Image
from collections import Counter
from multiprocessing import Pool
import os, sys, argparse, time, cv2

# python python/shoes/draw_bbox.py -i data/shoes/images -o data/shoes/draw_images

def get_file_list(input_root, output_root):
    #for root, dirs, files in os.walk(input_root):
    input_img_list = []
    output_img_list = []   
    if not os.path.exists(output_root):
        os.mkdir(output_root)     
    for temp_dir in os.listdir(input_root):
        input_dir = os.path.join(input_root, temp_dir)
        output_dir = os.path.join(output_root, temp_dir)
        if input_dir.endswith('.jpg'):
            input_img_list.append(input_dir)
            output_img_list.append(output_dir)
        elif os.path.isdir(input_dir):
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            for input_file in os.listdir(input_dir):
                if input_file.endswith('.jpg'):
                    input_img_list.append(os.path.join(input_dir, input_file))
                    output_img_list.append(os.path.join(output_dir, input_file))
        else:
            pass
    print len(input_img_list), len(output_img_list)
    return input_img_list, output_img_list


def draw_bbox(arguments):
    input_img_file, output_img_file = arguments[0], arguments[1]
    img = cv2.imread(input_img_file)
    grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayed = cv2.blur(grayed, (3, 3))
    width = grayed.shape[1]
    height = grayed.shape[0]
    canimg = cv2.Canny(grayed, 20, 80)

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
    
    print input_img_file,  'Area:', (right-left)*(bottom-top)

    #raw_img  = Image.open(input_img_file)
    #crop_img = raw_img.crop((left, top, right, bottom))
    #crop_img.save(output_img_file, quality=99)
    cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), thickness = 2)
    cv2.imwrite(output_img_file, img)

def get_args():
    parser = argparse.ArgumentParser(description = 'draw the bbox of images')
    parser.add_argument('-i', dest = 'input_root',
        help = 'input root of images', default = None, type = str)
    parser.add_argument('-o', dest = 'output_root', 
        help = 'output root of images', default = None, type = str)
    parser.add_argument('-c', dest = 'cpu_num', 
        help = 'cpu number for multiprocessing', default = 8, type = int)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    args = get_args()

    input_root  = args.input_root
    output_root = args.output_root
    cpu_num     = args.cpu_num

    tic = time.time()

    input_img_list, output_img_list = get_file_list(input_root, output_root)
    pool = Pool(cpu_num)
    pool.map(draw_bbox, zip(input_img_list, output_img_list))

    toc = time.time()
    print 'running time: {} seconds'.format(toc-tic)