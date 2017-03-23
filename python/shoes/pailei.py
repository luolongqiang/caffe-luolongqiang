# -*- coding:UTF-8 -*-
import cv2
import math
import numpy as np
from PIL import Image
from collections import Counter
from PIL import ImageEnhance
import Image, ImageFilter
import argparse
import os
import os.path as osp
import copy
import json


# img = cv2.imread('/wyh/privacy/works/201611/1121/abnormal/12.jpg')


def folderparse(imagefolder):
    for root, dirs, files in os.walk(imagefolder):

        lista = []
        for fn in files:

            if fn.endswith('.jpg') or fn.endswith('.jpeg') or fn.endswith('.bmp') or fn.endswith('.JPG') or fn.endswith(
                    '.png'):
                filename = osp.join(root, fn)
                # print filename
                lista.append(filename)

        listb = lista
        # print listb

        m = len(listb)
        sum = 0
        s = []
        for i in range(0, m):
            img = cv2.imread(listb[i])
            # img = fit_size(img, 300)
            grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            grayed = cv2.blur(grayed, (3, 3))
            w = grayed.shape[1]
            h = grayed.shape[0]
            canimg = cv2.Canny(grayed, 20, 80)

            if np.max(canimg) == 0:
                t = 0
                r = w - 1
                b = h - 1
                l = 0
            else:
                linepix = np.where(canimg == 255)
                t = min(linepix[0])
                r = max(linepix[1])
                b = max(linepix[0])
                l = min(linepix[1])

            imgx = Image.open(listb[i])
            imgy = imgx.crop((l, t, r, b))
            n = listb[i].split('/')


            # print n[-2]
            # outputfilename = listb[i] + '.jpg'

            path = '/wyh/privacy/works/201701/0104/ca_grab/3/'
            outputfilename = path + n[-1]  # n[-1]是文件的名称
            imgy.save(outputfilename, quality=99)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='imagefolder')
   # parser.add_argument('-o', dest='output')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    folderparse(args.imagefolder)#,args.output)