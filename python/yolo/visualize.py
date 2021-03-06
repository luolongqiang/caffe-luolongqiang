#*******************
#date:2016-12-14
#author:luolongqiang
#*******************

import numpy as np
import argparse
import cv2
import sys,os

#im_size=(512,768)
im_size=(767,512)

def get_labels(input_txt):
    labels = []
    with open(input_txt,'r') as fi:
        for line in fi:
            temp = map(float, line.strip().split(' '))
            labels.append(temp)
    return labels

def visualize_for_bbox(input_img, labels):
    im = cv2.imread(input_img)     
    for obj in labels:
        x, y, w, h = im_size[0]*obj[1], im_size[1]*obj[2], im_size[0]*obj[3], im_size[1]*obj[4]
        x1, y1 = int(x-w/2), int(y-h/2)
        x2, y2 = int(x+w/2), int(y+h/2)
        cv2.rectangle(im, (x1, y1), (x2, y2), (255, 0, 0), thickness = 2)
    return im

def get_args():
    parser = argparse.ArgumentParser(description='visualize for images')
    parser.add_argument('-img', dest='img', 
        help='input image', default=None, type=str)
    parser.add_argument('-txt', dest='txt', 
        help='input txt', default=None, type=str)
    parser.add_argument('-out', dest='out', 
        help='output image', default=None, type=str)   
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

if __name__=="__main__":
       
    args = get_args()
    img = args.img
    txt = args.txt
    out_path = args.out
    labels = get_labels(txt)
    out_img = visualize_for_bbox(img, labels)
    cv2.imwrite(out_path, out_img)

