# -*- coding:UTF-8 -*-
import numpy as np
from numpy import array
from pandas import DataFrame
from PIL import Image
import os, sys, time, argparse, shutil, math
from multiprocessing import Pool, cpu_count
import cv2, caffe
caffe.set_mode_cpu()
 
# python python/shoes/get_same_scale_img.py -m models/shoes_flickr/deploy.prototxt -w models/shoes_flickr/flickr_ft_out/_iter_1.6w.caffemodel -i data/shoes/images -o data/shoes/images/trans_images

scale_list = [0.75, 1.00, 0.84, 0.6, 0.84, 0.86, 1.06, 1.31, 0.6, 0.84]
class_num = len(scale_list)
transform_scale = np.zeros((class_num, class_num))
for i in range(class_num):
    for j in range(class_num):
        transform_scale[i, j] = scale_list[i]/scale_list[j]
esp_margin = 10

def get_file_list(input_root, output_root):
    #for root, dirs, files in os.walk(input_root):
    input_img_list = []
    output_img_list = []   
    if not os.path.exists(output_root):
        os.mkdir(output_root)     
    for temp_dir in os.listdir(input_root):
        input_dir = os.path.join(input_root, temp_dir)
        output_dir = os.path.join(output_root, temp_dir)
        if input_dir.endswith('.jpg'):
            input_img_list.append(input_dir)
            output_img_list.append(output_dir)
        elif os.path.isdir(input_dir):
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            for input_file in os.listdir(input_dir):
                if input_file.endswith('.jpg'):
                    input_img_list.append(os.path.join(input_dir, input_file))
                    output_img_list.append(os.path.join(output_dir, input_file))
        else:
            pass
    print len(input_img_list), len(output_img_list)
    return input_img_list, output_img_list

def make_prediction(img_file_name):
    img = caffe.io.load_image(img_file_name)  # load the image using caffe io
    width, height = net.blobs['data'].data.shape[2:]
    inputs = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    # mu = array([182, 186, 186])
    # transformer.set_mean('data', mu)
    transformer.set_raw_scale('data', 256.0) 
    transformer.set_channel_swap('data', (2,1,0))

    out = net.forward(data = np.asarray([transformer.preprocess('data', inputs)]))
    probs = out['prob'][0]
    pred_label = np.argmax(probs) + 1
    return pred_label

def crop_bbox(img):
    grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayed = cv2.blur(grayed, (3, 3))
    width = grayed.shape[1]
    height = grayed.shape[0]
    canimg = cv2.Canny(grayed, 10, 20)

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
    
    esp = 10
    if top-esp>=0 and left-esp>=0 and bottom+esp<=height and right+esp<=width:
        top = top-esp
        left = left-esp
        bottom = bottom+esp
        right = right+esp
    crop_img = img[top:bottom, left:right, :]
    bbox_area = (right-left)*(bottom-top)

    return bbox_area, crop_img

def transform_image(input_img_list, output_img_list):
    pred_label_list = []
    bbox_area_list = []
    crop_img_list = []
    max_margin = 0
    for img_file_name in input_img_list:
        pred_label = make_prediction(img_file_name)
        pred_label_list.append(pred_label)
        img = cv2.imread(img_file_name)
        bbox_area, crop_img = \
           crop_bbox(img[esp_margin:-esp_margin, esp_margin:-esp_margin, :])
        bbox_area_list.append(bbox_area)
        crop_img_list.append(crop_img)
        max_margin = max(max_margin, img.shape[0], img.shape[1])

    base_index = np.argsort(bbox_area_list)[len(input_img_list)*2/3]
    base_label = pred_label_list[base_index]
    base_img_file = input_img_list[base_index]
    base_area = bbox_area_list[base_index]
    base_crop_img = crop_img_list[base_index]
        
    print base_img_file, base_label, base_area

    for i, input_img_file in enumerate(input_img_list):      
        output_img_file = output_img_list[i]  
        if input_img_file == base_img_file:
            get_dest_img(base_crop_img, max_margin, output_img_file)
        else:
            pred_label = pred_label_list[i]
            raw_area = bbox_area_list[i]
            crop_img = crop_img_list[i]
            scale = transform_scale[pred_label-1, base_label-1]
            target_area = base_area*scale
            print input_img_file, pred_label, raw_area , target_area

            raw_crop_width, raw_crop_height = crop_img.shape[1], crop_img.shape[0]
            zoom_scale = math.sqrt(target_area*1.0/raw_area)
            target_crop_width = int(raw_crop_width*zoom_scale)
            target_crop_height = int(raw_crop_height*zoom_scale)
            target_crop_img = cv2.resize(crop_img, (target_crop_width, \
               target_crop_height), interpolation=cv2.INTER_LINEAR)
            
            get_dest_img(target_crop_img, max_margin, output_img_file)                                                                                           
    #end

def get_dest_img(crop_img, max_margin, output_img_file):
    
    crop_width, crop_height = crop_img.shape[1],  crop_img.shape[0]
    max_margin = max(max_margin, crop_width, crop_height)
    lr_dist = max_margin-crop_width
    tb_dist = max_margin-crop_height

    dest_img = np.ones((max_margin, max_margin, 3), np.uint8)*255 # crop_img[0, 0, 0]
    dest_img[tb_dist/2:tb_dist/2+crop_height, lr_dist/2:lr_dist/2+crop_width, :] = crop_img
    cv2.imwrite(output_img_file, dest_img)

def get_args():
    parser = argparse.ArgumentParser(description = 'transform scale for images')
    parser.add_argument('-m', dest = 'model', 
        help = 'network with format .prototxt', default = None, type = str)
    parser.add_argument('-w', dest = 'weight',
        help = 'trained weights with format .caffemodel', default = None, type = str)
    parser.add_argument('-i', dest = 'input_root',
        help = 'input root of images', default = None, type = str)
    parser.add_argument('-o', dest = 'output_root', 
        help = 'output root of images', default = None, type = str)
    parser.add_argument('-n', dest = 'class_num', 
        help = 'class number', default = 10, type = int)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args    

if __name__ == '__main__':

    global net  

    args = get_args()

    model_file  = args.model
    weight_file = args.weight
    input_root  = args.input_root
    output_root = args.output_root
    class_num   = args.class_num

    tic  = time.time()

    net  = caffe.Net(model_file, weight_file, caffe.TEST)

    input_img_list, output_img_list = get_file_list(input_root, output_root)
    transform_image(input_img_list, output_img_list)
     
    toc  = time.time()
    print 'running time:{:.3f} minutes'.format((toc - tic)/60.0)
    
