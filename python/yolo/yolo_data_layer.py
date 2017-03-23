import sys
import os.path as osp
this_dir = osp.dirname(__file__)
caffe_path = osp.join(this_dir, '..')
sys.path.insert(0, caffe_path)
from datetime import datetime

import caffe
import numpy as np
import random
import cv2


class YoloDataLayer(caffe.Layer):

    def setup(self, bottom, top):
        params = eval(self.param_str)  # yolo_small_train_vel.prototxt/python_param/param_str 
        self.input_txt = params['input_file']
        self.batch_size = params['batch_size']
        self.classes = params['classes']
        self.side = params['side']
        self.jitter = params['jitter']
        self.width = params['width']
        self.height = params['height']

        file_list = open(self.input_txt).read().splitlines()
        random.shuffle(file_list)
        self.file_list = file_list 
        self.idx = 0

        if len(top) != 2:
            raise Exception('Need to define two tops.')
        if len(bottom) != 0:
            raise Exception('Do not define a bottom')
    
    def reshape(self, bottom, top):
        channels = 3
        top[0].reshape(self.batch_size, channels, self.width, self.height)
        top[1].reshape(self.batch_size, self.side*self.side*(5+self.classes))
        for i in range(0, self.batch_size):
            self.data, self.truth = self.load_yolo_image(self.file_list[self.idx], self.width, self.height, self.side,
                                                         self.classes, self.jitter)
            if self.idx + 1 == len(self.file_list):
                self.idx = 0
                random.shuffle(self.file_list)
            else:
                self.idx += 1
            top[0].data[i, ...] = self.data
            top[1].data[i, ...] = self.truth
        
    def forward(self, bottom, top):
        pass

    def backward(self, bottom, top):
        pass

    def load_yolo_image1(self, filename, width, height, side, classes, jitter):
        
        origin_img = cv2.imread(filename, 1)
        float_img = origin_img/255.0
        
        origin_height, origin_width = origin_img.shape
        width_offset = int(origin_width*jitter)
        height_offset = int(origin_height*jitter)
        
        left_offset = random.randint(-width_offset, 0)
        right_offset = random.randint(-width_offset, 0)
        top_offset = random.randint(-height_offset, 0)
        bottom_offset = random.randint(-height_offset, 0)

        crop_width = origin_width - left_offset - right_offset
        crop_height = origin_height - top_offset - bottom_offset
        scale_x = float(crop_width)/origin_width
        scale_y = float(crop_height)/origin_height
        scale_x_off = float(left_offset)/crop_width
        scale_y_off = float(top_offset)/crop_height

        crop_img = np.zeros((crop_height, crop_width, 3), dtype='float32')

        src_left = left_offset if left_offset > 0 else 0
        src_top = top_offset if top_offset > 0 else 0
        src_right = origin_width-right_offset-1 if right_offset > 0 else origin_width-1
        src_bottom = origin_height-bottom_offset-1 if bottom_offset > 0 else origin_height-1
        dst_left = 0 if left_offset > 0 else -left_offset
        dst_top = 0 if top_offset > 0 else -top_offset
        dst_right = crop_width-1 if right_offset > 0 else crop_width+right_offset-1
        dst_bottom = crop_height-1 if bottom_offset > 0 else crop_height+bottom_offset-1
        crop_img[dst_top: dst_bottom, dst_left: dst_right, :] = float_img[src_top: src_bottom, src_left: src_right, :]
        
        resize_img = cv2.resize(crop_img, (width, height), interpolation=cv2.INTER_LINEAR)

        img_data = np.array(resize_img, dtype=np.float32)
        img_data = img_data[:, :, ::-1]
        img_data = img_data.transpose((2, 0, 1))

        box_file = filename.replace('.jpg', '.txt')
        box_file = box_file.replace('JPEGImages', 'labels')
        yolo_truth = np.zeros((side*side*(5+classes)), dtype='float32')
        for line in open(box_file).read().splitlines():
            label, x, y, w, h = map(float, line.split())

            left = x - w/2.0
            right = x + w/2.0
            top = y - h/2.0
            bottom = y + h/2.0

            left = self.constrain(0.0, 1.0, left / scale_x - scale_x_off)
            right = self.constrain(0.0, 1.0, right / scale_x - scale_x_off)
            top = self.constrain(0.0, 1.0, top / scale_y - scale_y_off)
            bottom = self.constrain(0.0, 1.0, bottom / scale_y - scale_y_off)
            
            x = (left + right) / 2.0
            y = (top + bottom) / 2.0
            w = right - left
            h = bottom - top

            if w < .01 or h < .01:
                continue

            col = int(x * side)
            row = int(y * side)
            x = x * side - col
            y = y * side - row
            index = (col + row * side) * (5+classes)
            yolo_truth[index] = 1
            yolo_truth[index+1+int(label)] = 1
            yolo_truth[index+1+classes] = x 
            yolo_truth[index+1+classes+1] = y
            yolo_truth[index+1+classes+2] = w
            yolo_truth[index+1+classes+3] = h

        return img_data, yolo_truth

    def constrain(self, low, high, x):
        if x < low:
            return low
        elif x > high:
            return high
        return x


