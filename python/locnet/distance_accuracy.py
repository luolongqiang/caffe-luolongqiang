import sys
import os.path as osp
import numpy as np
this_dir = osp.dirname(__file__)
caffe_path = osp.join(this_dir, '..')
sys.path.insert(0, caffe_path)

import caffe


class DistanceAccuracyLayer(caffe.Layer):

    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception('Need to define two bottom')
    
    def reshape(self, bottom, top):
        top[0].reshape(4)
        # top[1].reshape(4)

    def forward(self, bottom, top):
        max_accu = np.zeros(4, dtype='float32')
        combine_accu = np.zeros(4, dtype='float32')
        # print bottom[0].shape[0]
        # print bottom[1].shape[0]
        # assert(bottom[0].shape == bottom[1].shape)
        num = bottom[0].num
        # print bottom[0].num, bottom[0].channels, bottom[0].height, bottom[0].width
        for i in range(0, num):
            pred_probs = bottom[0].data[i][0]
            target_dis = bottom[1].data[i][0]

            max_accu[0] += np.fabs(np.argmax(pred_probs[0]) - np.argmax(target_dis[0]))
            max_accu[1] += np.fabs(np.argmax(pred_probs[1]) - np.argmax(target_dis[1]))
            max_accu[2] += np.fabs(np.argmax(pred_probs[3]) - np.argmax(target_dis[3]))
            max_accu[3] += np.fabs(np.argmax(pred_probs[4]) - np.argmax(target_dis[4]))

            left, right = decode_loc_prob(pred_probs[:3])
            top, bottom = decode_loc_prob(pred_probs[3:])
            combine_accu[0] += np.fabs(left - np.argmax(target_dis[0]))
            combine_accu[1] += np.fabs(right - np.argmax(target_dis[1]))
            combine_accu[2] += np.fabs(top - np.argmax(target_dis[3]))
            combine_accu[3] += np.fabs(bottom - np.argmax(target_dis[4]))

        max_accu /= num
        combine_accu /= num
        # top[0].data[...] = max_accu
        top[0].data[...] = combine_accu

    def backward(self, bottom, top):
        pass


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
    return border_start, border_end
