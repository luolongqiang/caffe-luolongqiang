import os.path as osp
import sys
import numpy as np
import random
import cv2

this_dir = osp.dirname(__file__)
caffe_path = osp.join(this_dir, '..')
sys.path.insert(0, caffe_path)

import caffe


class LocNetDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.input_txt = params['input_file']
        self.batch_size = params['batch_size']
        # self.classes = params['classes']
        self.jitter = params['jitter']
        self.width = params['width']
        self.height = params['height']
        self.resolution = params['resolution']

        file_list = open(self.input_txt).read().splitlines()
        random.shuffle(file_list)
        self.file_list = file_list
        self.idx = 0

        if len(top) != 3:
            raise Exception('Need to define three tops.')
        if len(bottom) != 0:
            raise Exception('Do not define a bottom')

    # top[0] -- data blob[N * C * W * H]
    # top[1] -- ground truth[N * 1 * 6 * resolution]
    # top[2] -- weights [N * 1 * 6 * resolution]
    def reshape(self, bottom, top):
        top[0].reshape(self.batch_size, 3, self.width, self.height)
        top[1].reshape(self.batch_size, 1, 6, self.resolution)
        top[2].reshape(self.batch_size, 1, 6, self.resolution)

        for i in range(0, self.batch_size):
            self.data, self.truth, self.weights = self.load_input(self.file_list[self.idx], self.width, self.height,
                                                                  self.resolution, self.jitter)
            if self.idx + 1 == len(self.file_list):
                self.idx = 0
                random.shuffle(self.file_list)
            else:
                self.idx += 1
            top[0].data[i, ...] = self.data
            top[1].data[i, 0, ...] = self.truth
            top[2].data[i, 0, ...] = self.weights

    def forward(self, bottom, top):
        pass

    def backward(self, bottom, top):
        pass

    def load_input(self, filename, width, height, resolution, jitter):

        origin_img = cv2.imread(filename, 1)
        float_img = origin_img / 255.0

        origin_width = float_img.shape[1]
        origin_height = float_img.shape[0]
        width_offset = int(origin_width * self.jitter)
        height_offset = int(origin_height * self.jitter)

        left_offset = random.randint(-width_offset, 0)
        right_offset = random.randint(-width_offset, 0)
        top_offset = random.randint(-height_offset, 0)
        bottom_offset = random.randint(-height_offset, 0)

        crop_width = origin_width - left_offset - right_offset
        crop_height = origin_height - top_offset - bottom_offset
        scale_x = float(crop_width) / origin_width
        scale_y = float(crop_height) / origin_height
        scale_x_off = float(left_offset) / crop_width
        scale_y_off = float(top_offset) / crop_height

        crop_img = np.zeros((crop_height, crop_width, 3), dtype='float32')
        src_left = left_offset if left_offset > 0 else 0
        src_top = top_offset if top_offset > 0 else 0
        src_right = origin_width - right_offset - 1 if right_offset > 0 else origin_width - 1
        src_bottom = origin_height - bottom_offset - 1 if bottom_offset > 0 else origin_height - 1
        dst_left = 0 if left_offset > 0 else -left_offset
        dst_top = 0 if top_offset > 0 else -top_offset
        dst_right = crop_width - 1 if right_offset > 0 else crop_width + right_offset - 1
        dst_bottom = crop_height - 1 if bottom_offset > 0 else crop_height + bottom_offset - 1
        crop_img[dst_top: dst_bottom, dst_left: dst_right, :] = float_img[src_top: src_bottom, src_left: src_right, :]

        resize_img = cv2.resize(crop_img, (width, height), interpolation=cv2.INTER_LINEAR)

        img_data = np.array(resize_img, dtype=np.float32)
        img_data = img_data[:, :, ::-1]
        img_data = img_data.transpose((2, 0, 1))

        # temp_img = resize_img*255
        box_file = filename.replace('.jpg', '.txt')
        box_file = box_file.replace('JPEGImages', 'labels')
        combined_truth = np.zeros((6, resolution), dtype='float32')

        for line in open(box_file).read().splitlines():
            label, x, y, w, h = int(line.split()[0]), float(line.split()[1]), float(line.split()[2]), float(
                line.split()[3]), float(line.split()[4])
            if label != 0:
                continue
            left = x - w / 2.0
            right = x + w / 2.0
            top = y - h / 2.0
            bottom = y + h / 2.0

            left = self.constrain(0.0, 1.0, left / scale_x - scale_x_off)
            right = self.constrain(0.0, 1.0, right / scale_x - scale_x_off)
            top = self.constrain(0.0, 1.0, top / scale_y - scale_y_off)
            bottom = self.constrain(0.0, 1.0, bottom / scale_y - scale_y_off)

            combined_truth, loss_weight = encode_coordinates(left, right, top, bottom, resolution)
        # basename = osp.basename(filename)
        # cv2.imwrite(osp.join('/home/zhangzhang/data/yolo/temp', basename), temp_img)
        return img_data, combined_truth, loss_weight

    def constrain(self, low, high, x):
        if x < low:
            return low
        elif x > high:
            return high
        return x


def encode_coordinates(left, right, top, bottom, resolution):
    combined_truth = np.zeros((6, resolution), dtype='float32')
    loss_weight = np.ones((6, resolution), dtype='float32')
    neg_loss = resolution * 0.5 / (resolution - 1.0)
    posi_loss = (resolution - 1.0) * neg_loss

    left = int(left * (resolution - 1) + 0.5)
    right = int(right * (resolution - 1) + 0.5)
    top = int(top * (resolution - 1) + 0.5)
    bottom = int(bottom * (resolution - 1) + 0.5)

    combined_truth[0][left] = 1.0
    combined_truth[1][right] = 1.0
    combined_truth[2][left: right + 1] = 1.0
    combined_truth[3][top] = 1.0
    combined_truth[4][bottom] = 1.0
    combined_truth[5][top: bottom + 1] = 1.0

    for i in [0, 1, 3, 4]:
        for j in range(0, resolution):
            if combined_truth[i][j] == 1:
                loss_weight[i][j] *= posi_loss
            if combined_truth[i][j] == 0:
                loss_weight[i][j] *= neg_loss

    return combined_truth, loss_weight
