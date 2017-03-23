import sys
import os.path as osp
import numpy as np
import cv2
import argparse
import math

this_dir = osp.dirname(__file__)
caffe_path = osp.join(this_dir, '..')
sys.path.insert(0, caffe_path)
import caffe

caffe.set_mode_gpu()


def deploy(image_file, model_file, weight_file):
    net = caffe.Net(model_file, weight_file, caffe.TEST)
    img = caffe.io.load_image(image_file)
    inputs = img
    src_width, src_height = img.shape[1], img.shape[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    out = net.forward_all(data=np.asarray([transformer.preprocess('data', inputs)]))

    result_blob = out['preds_loc']
    resolution = out['preds_loc'].shape[3]
    assert (out['preds_loc'].shape[2] == 6)
    probs_x = np.reshape(result_blob[0][0][:3], (3, resolution))
    x_start, x_end = decode_loc_prob(probs_x)
    probs_y = np.reshape(result_blob[0][0][3:], (3, resolution))
    y_start, y_end = decode_loc_prob(probs_y)
    left, right = decode_coordinates(x_start, x_end, resolution, src_width)
    top, bottom = decode_coordinates(y_start, y_end, resolution, src_height)

    base_name = osp.basename(image_file)
    origin_img = cv2.imread(image_file, 1)
    cv2.rectangle(origin_img, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.imwrite(osp.join('/home/zhangzhang/data/temp', base_name), origin_img)

    return left, right, top, bottom


def decode_coordinates(start, end, src_resolution, dst_resolution):
    scale = 1.0/src_resolution
    new_start = int(start*scale*dst_resolution)
    new_end = int(end*scale*dst_resolution)
    return new_start, new_end


def decode_loc_prob(probs):
    # shape of probs is (3, resolution)
    assert (probs.shape[0] == 3)
    min_prob = 0.0001
    positive_probs = np.maximum(probs, min_prob)
    negative_probs = np.maximum(1 - probs, min_prob)

    start_probs = -np.log(positive_probs[0])
    end_probs = -np.log(positive_probs[1])
    inside_probs = -np.log(positive_probs[2])

    non_start_probs = -np.log(negative_probs[0])
    non_end_probs = -np.log(negative_probs[1])
    outside_probs = -np.log(negative_probs[2])

    inside_probs_sum = np.cumsum(np.insert(inside_probs, 0, 0))
    outside_probs_sum = np.cumsum(np.insert(outside_probs, 0, 0))

    resolution = probs.shape[1]
    min_negative_likelihood = 10000
    border_start = 0
    border_end = 0
    for start in range(resolution - 1):
        for end in range(start + 1, resolution):
            # print start, end
            negative_likelihood = (inside_probs_sum[end + 1] - inside_probs_sum[start]) - \
                                  (outside_probs_sum[end + 1] - outside_probs_sum[start]) + \
                                  (start_probs[start] + end_probs[end]) - \
                                  (non_start_probs[start] + non_end_probs[end])
            if negative_likelihood < min_negative_likelihood:
                min_negative_likelihood = negative_likelihood
                border_start = start
                border_end = end
    prob = probs[0][border_start] * probs[1][border_end]
    print prob
    return border_start, border_end


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', dest='image_list_file')
    parser.add_argument('-m', dest='model_file')
    parser.add_argument('-w', dest='weights_file')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    f_list = open(args.image_list_file).read().split()
    for image_file in f_list:
        deploy(image_file, args.model_file, args.weights_file)

