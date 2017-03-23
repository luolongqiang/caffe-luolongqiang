import sys
import os.path as osp
import numpy as np
import random
import cv2
import caffe

this_dir = osp.dirname(__file__)
caffe_path = osp.join(this_dir, '..')
sys.path.insert(0, caffe_path)
#sys.path.insert(0, '/home/luolongqiang/deepdraw/caffe/python/deepFashion/')

class CategoryLabelDataLayerSync(caffe.Layer):

    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.input_txt = params['input_file']
        self.batch_size = params['batch_size']
        self.class_num = params['class_num']
        self.width = params['width']
        self.height = params['height']
        self.label_list = ['3', '6', '11', '16', '17', '18', '19', '32', '33', '41']

        file_list = open(self.input_txt).read().splitlines()
        random.shuffle(file_list)
        self.file_list = file_list 
        self.idx = 0

        if len(top) != 2:
            raise Exception('Need to define two tops.')
        if len(bottom) != 0:
            raise Exception('Do not define a bottom')
    
    def reshape(self, bottom, top):
        top[0].reshape(self.batch_size, 3, self.width, self.height)
        top[1].reshape(self.batch_size, self.class_num)
        for i in range(0, self.batch_size):
            self.img_data, self.label = self.load_yolo_image(self.file_list[self.idx], self.width, self.height)
            if self.idx == len(self.file_list) - 1:
                self.idx = 0
                random.shuffle(self.file_list)
            else:
                self.idx += 1
            top[0].data[i, ...] = self.img_data
            top[1].data[i, ...] = self.label
        
    def forward(self, bottom, top):
        pass

    def backward(self, bottom, top):
        pass

    def load_yolo_image(self, file_label, width, height):
        file_name = file_label.strip().split()[0]
        origin_img = cv2.imread(file_name, 1)
        resize_img = cv2.resize(origin_img/255.0, (width, height), interpolation=cv2.INTER_LINEAR)
        img_data = np.array(resize_img, dtype=np.float32)
        img_data = img_data[:, :, ::-1]
        img_data = img_data.transpose((2, 0, 1))

        label_dict = dict(zip(self.label_list, np.identity(len(label_list))))
        origin_label = file_label.strip().split()[-1]
        label = label_dict[origin_label]

        return img_data, label
