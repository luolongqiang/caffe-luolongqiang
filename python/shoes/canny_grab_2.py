# -*- coding:utf-8 -*-
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
e1 = cv2.getTickCount()

def folderparse(imagefolder):
    for root, dirs, files in os.walk(imagefolder):
        for fn in files:
            if fn.endswith('.jpg') or fn.endswith('.jpeg') or fn.endswith('.bmp')or fn.endswith('.JPG')or fn.endswith('.png'):
                filename = osp.join(root,fn)
                print filename
                img = cv2.imread(filename)
                if img == None:
                    continue
#img = Image.open('/wyh/privacy/works/201611/1121/image/21.jpg')
#img1 = ImageEnhance.Color(img).enhance(8.0)#Color\Contrast
                #img1 = ImageEnhance.Sharpness(img).enhance(4) #range=[0,4],step=0.5
                img = fit_size(img, 300)
                grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                    
                grayed = cv2.blur(grayed, (5, 5))
                w = grayed.shape[1]
                h = grayed.shape[0]
                canimg = cv2.Canny(grayed, 30, 90)                                                  
                #cannyfile = .canny.bmp'
                #cv2.imshow('2',canimg)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(199, 199)) #ELLIPSE   _RECT
                #th3 = cv2.morphologyEx(canimg, cv2.MORPH_CLOSE, kernel)
                th3 = cv2.dilate(canimg,kernel)
                #cv2.bitwise_and(th3,th3,mask=mask2)
                #cv2.imshow('2',th3)
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
                
                mask = np.zeros(img.shape[:2],np.uint8)
                bgdModel = np.zeros((1,65),np.float64)
                fgdModel = np.zeros((1,65),np.float64)
                rect = (l,t,r-l,b-t)
                # 函数的返回值是更新的 mask, bgdModel, fgdModel
                cv2.grabCut(img,mask,rect,bgdModel,fgdModel,3,cv2.GC_INIT_WITH_RECT)
                mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
                img = img*mask2[:,:,np.newaxis]
                #print img
                output = cv2.bitwise_and(th3,th3,mask=mask2)
                #output1 = copy.deepcopy(output)
                y,x =output.shape
                sum=0
                x_sum=0
                y_sum=0
                for i in range(0,y):
                    for j in range(0,x):
                        if output[i,j]==255:
                            sum+=1
                            x_sum=x_sum+j
                            y_sum=y_sum+i
                        else:
                            pass
                
                #print sum
                x_mean = x_sum/sum
                y_mean = y_sum/sum
                
                focus=(x_mean,y_mean)
                #print focus
                #cv2.line(img,focus,focus,(255,0,255),15) 
                outputfilename = fn + '.jpg'
                maskfilename = fn + '.bmp'
                outputfolder = '/wyh/privacy/works/201611/1121/shoes1/' #查询出来的图片放置位置
                outputfolder1=outputfolder+outputfilename
                outputfolder2=outputfolder+maskfilename
                cv2.imwrite(outputfolder1,img) 
                cv2.imwrite(outputfolder2,output)
    
 

                
e2 = cv2.getTickCount()
time = (e2-e1)/cv2.getTickFrequency()
print time
#cv2.imshow('y', img2)
#cv2.imwrite('/wyh/privacy/works/201611/1121/1/12.jpg', img)
            #cv2.waitKey(0)
def fit_size(img, size):
    img_w = img.shape[1]
    img_h = img.shape[0]
    
    assert img_w > 0
    assert img_h > 0 
    
    assert isinstance(size, int)      
    assert size > 0 
    
    if img_w >= img_h:
        scale = float(size)/float(img_h)
        fit_w = max(int(scale*img_w),1)   
        fit_h = size
    else:
        scale = float(size)/float(img_w)
        fit_w = size
        fit_h = max(int(scale*img_h),1)  
        
    dim = (fit_w,fit_h)
    resized_img = cv2.resize(img, dim, interpolation = cv2.INTER_LINEAR)
    return resized_img#,scale 

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

 
