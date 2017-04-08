# -*- coding:utf-8 -*-

# python python/shoes/transparency.py

"""本程序可以改变png透明背景的颜色，只要事先把抠出来的图片保存为png格式，随后便可以用这个程序实现背景的变换"""
import cv2
import numpy as np 
import PIL 
from PIL import Image
im = Image.open('data/shoes/t1.png')
im1 = Image.open('data/shoes/t1.bmp')
#im.save('3.net.png')
im = im.convert("RGBA")
x, y = im.size 
print im.mode
#try: 
    # 使用白色来填充背景 from：www.jb51.net
    # (alpha band as paste mask). 
#p = Image.new('RGBA', im.size, (0,0,255))
p = Image.open('data/shoes/background/2.jpg')#p是用来充当修图之后的背景的。
p = p.convert("RGBA")
p = p.resize(im.size)#转换成与im同等大小的
p.paste(im, (0, 0, x, y), im1)
p.save('data/shoes/t2.png')
#except:
#    pass
print 'over' # using for remind
