import sys
import os.path as osp
import numpy as np
import random
import cv2
import caffe

this_dir = osp.dirname(__file__)
caffe_path = osp.join(this_dir, '..')
sys.path.insert(0, caffe_path)
sys.path.insert(0, '/home/luolongqiang/deepdraw/caffe/python/product/')

class MultiTaskDataLayerSync(caffe.Layer):

    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.input_txt = params['input_file']
        self.batch_size = params['batch_size']
        self.width = params['width']
        self.height = params['height']

        file_list = open(self.input_txt, 'r').read().splitlines()
        random.shuffle(file_list)
        self.file_list = file_list
        self.idx = 0

        if len(top) != 5:
            raise Exception('Need to define five tops.')
        if len(bottom) != 0:
            raise Exception('Do not define a bottom')
    
    def reshape(self, bottom, top):
        channels = 3
        top[0].reshape(self.batch_size, channels, self.width, self.height)
        top[1].reshape(self.batch_size, 1)
        top[2].reshape(self.batch_size, 1)
        top[3].reshape(self.batch_size, 1)
        top[4].reshape(self.batch_size, 1)
        for i in range(0, self.batch_size):
            self.data, self.labels = self.load_yolo_image(self.file_list[self.idx], self.width, self.height)
            if self.idx + 1 == len(self.file_list):
                self.idx = 0
                random.shuffle(self.file_list)
            else:
                self.idx += 1
            top[0].data[i, ...] = self.data
            top[1].data[i, ...] = self.labels[0]
            top[2].data[i, ...] = self.labels[1]
            top[3].data[i, ...] = self.labels[2]
            top[4].data[i, ...] = self.labels[3]
        
    def forward(self, bottom, top):
        pass

    def backward(self, bottom, top):
        pass

    def load_yolo_image(self, file_label, width, height):
        line_list = file_label.strip().split()
        file_name = line_list[0]
        labels = map(int, line_list[1:])

        img_data = cv2.imread(file_name, 1)
        img_data = cv2.resize(img_data/256.0, (width, height), interpolation=cv2.INTER_LINEAR)
        img_data = np.array(img_data, dtype=np.float32)
        img_data = img_data[:, :, ::-1]
        img_data = img_data.transpose((2, 0, 1))

        return img_data, labels

