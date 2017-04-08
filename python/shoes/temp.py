# -*- coding:UTF-8 -*-
import numpy as np
from numpy import array 
from PIL import Image
from multiprocessing import Pool
import os, sys, argparse, time
import caffe, cv2
caffe.set_mode_cpu()

# python python/shoes/temp.py -i data/shoes/362-test/detection_error/3-1.jpg -o data/shoes/362-test/detection_error/t3.jpg -t yolo -m models/shoes_demo/detection_deploy.prototxt -w models/shoes_demo/shoes_detection.caffemodel

def get_bbox_by_canny(input_img_file, output_img_file):
    img = cv2.imread(input_img_file)
    grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayed = cv2.GaussianBlur(grayed, (7, 7), sigmaX=0, sigmaY=0)
    #grayed = cv2.blur(grayed, (3, 3))
    width  = grayed.shape[1]
    height = grayed.shape[0]
    canimg = cv2.Canny(grayed, 30, 90)

    if np.max(canimg) == 0:
        top    = 0
        right  = width - 1
        bottom = height - 1
        left   = 0
    else:
        linepix = np.where(canimg == 255)
        top    = min(linepix[0])
        right  = max(linepix[1])
        bottom = max(linepix[0])
        left   = min(linepix[1])
    
    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.rectangle(img, (left, top - 20), (right, top), (125, 125, 125), -1)
    cv2.putText(img, 'canny', (left+5, top-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.imwrite(output_img_file, img)

def get_bbox_by_yolo(model_file, weight_file, input_img_file, output_img_file):
    net = caffe.Net(model_file, weight_file, caffe.TEST)
    inputs = caffe.io.load_image(input_img_file) # RGB and /255
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    out = net.forward_all(data = np.asarray([transformer.preprocess('data', inputs)]))
    img = cv2.imread(input_img_file)
    top, right, bottom, left, prob = interpret_output(out['result'][0], img.shape[1], img.shape[0])
    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.rectangle(img, (left, top - 20), (right, top), (125, 125, 125), -1)
    cv2.putText(img, 'yolo', (left+5, top-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.imwrite(output_img_file, img)

def interpret_output(output, w_img, h_img):
    num_box = 2
    num_class = 1
    grid_size = 7
    iou_threshold = 0.5
    index1 = grid_size**2*num_class
    index2 = index1+grid_size**2*num_box
    probs = np.zeros((grid_size, grid_size, num_box, num_class))
    class_probs = np.reshape(output[0:index1], (grid_size, grid_size, num_class))
    scales = np.reshape(output[index1:index2], (grid_size, grid_size, num_box))
    boxes = np.reshape(output[index2:], (grid_size, grid_size, num_box, 4))
    offset = np.reshape(np.array([np.arange(grid_size)]*(grid_size*num_box)), \
       (num_box, grid_size, grid_size))
    offset = np.transpose(offset, (1, 2, 0))

    # ------- get bbox -----------
    boxes[:, :, :, 0] += offset
    boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
    boxes[:, :, :, 0:2] = boxes[:, :, :, 0:2] / (grid_size*1.0)
    boxes[:, :, :, 2] = np.multiply(boxes[:, :, :, 2], boxes[:, :, :, 2])
    boxes[:, :, :, 3] = np.multiply(boxes[:, :, :, 3], boxes[:, :, :, 3])

    boxes[:, :, :, 0] *= w_img
    boxes[:, :, :, 1] *= h_img
    boxes[:, :, :, 2] *= w_img
    boxes[:, :, :, 3] *= h_img

    # ------ get confidence ------
    for i in range(num_box):
        for j in range(num_class):
            probs[:, :, i, j] = np.multiply(class_probs[:, :, j], scales[:, :, i])

    # --- confidence threshold ---
    filter_mat_probs = np.array(probs >= np.max(probs), dtype='bool')
    filter_mat_boxes = np.nonzero(filter_mat_probs)
    boxes_filtered = boxes[filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]
    probs_filtered = probs[filter_mat_probs]

    argsort = np.array(np.argsort(probs_filtered))[::-1]
    boxes_filtered = boxes_filtered[argsort][0]
    prob = probs_filtered[argsort][0]
    
    # get top, right, bottom, left
    x, y, w, h = tuple(boxes_filtered)
    left   = int(x - w/2)
    top    = int(y - h/2)
    right  = int(x + w/2)
    bottom = int(y + h/2)
    if left < 0:
        left = 0
    if top < 0:
        top = 0
    if right > w_img:
        right = w_img
    if bottom > h_img:
        bottom = h_img

    return top, right, bottom, left, prob


def get_args():
    parser = argparse.ArgumentParser(description = 'get shoes data')
    parser.add_argument('-i', dest = 'input_img',
        help = 'input image', default = None, type = str) 
    parser.add_argument('-o', dest = 'output_img', 
        help = 'output image', default = None, type = str) 
    parser.add_argument('-t', dest = 'type', 
        help = 'canny or yolo', default = 'canny', type = str) 
    parser.add_argument('-m', dest='model', 
        help='network with format .prototxt', default=None, type=str)
    parser.add_argument('-w', dest='weight',
        help='weights with format .caffemodel', default=None, type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    args = get_args()

    input_img_file  = args.input_img
    output_img_file = args.output_img

    tic = time.time()
    
    if args.type == 'canny':
        get_bbox_by_canny(input_img_file, output_img_file)
    else:
        model_file  = args.model
        weight_file = args.weight
        get_bbox_by_yolo(model_file, weight_file, input_img_file, output_img_file)
        
    toc = time.time()
    print 'running time: {} seconds'.format(toc-tic)