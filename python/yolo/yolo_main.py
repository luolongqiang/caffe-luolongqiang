import os, sys, time
import cv2
import argparse
import caffe
import numpy as np
from multiprocessing import Pool, cpu_count

def interpret_output(output, img_width, img_height):
    classes = ['body', 'face']
    w_img = img_width
    h_img = img_height
    global prob_threshold
    iou_threshold = 0.3
    num_class = 2
    num_box = 2
    grid_size = 7
    index1 = grid_size**2*num_class
    index2 = index1+grid_size**2*num_box
    probs = np.zeros((grid_size, grid_size, num_box, num_class))
    class_probs = np.reshape(output[0:index1], (grid_size, grid_size, num_class))
    scales = np.reshape(output[index1:index2], (grid_size, grid_size, num_box))
    boxes = np.reshape(output[index2:], (grid_size, grid_size, num_box, 4))
    offset = np.transpose(np.reshape(np.array([np.arange(grid_size)]*(grid_size*num_box)), (num_box, grid_size, grid_size)), (1, 2, 0))

    boxes[:, :, :, 0] += offset
    boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
    boxes[:, :, :, 0:2] = boxes[:, :, :, 0:2] / (grid_size*1.0)
    boxes[:, :, :, 2] = np.multiply(boxes[:, :, :, 2], boxes[:, :, :, 2])
    boxes[:, :, :, 3] = np.multiply(boxes[:, :, :, 3], boxes[:, :, :, 3])

    boxes[:, :, :, 0] *= w_img
    boxes[:, :, :, 1] *= h_img
    boxes[:, :, :, 2] *= w_img
    boxes[:, :, :, 3] *= h_img

    for i in range(num_box):
        for j in range(num_class):
            probs[:, :, i, j] = np.multiply(class_probs[:, :, j], scales[:, :, i])
            # probs[:,:,i,j] = scales[:,:,i]

    filter_mat_probs = np.array(probs >= prob_threshold, dtype='bool')
    filter_mat_boxes = np.nonzero(filter_mat_probs)
    boxes_filtered = boxes[filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]
    probs_filtered = probs[filter_mat_probs]
    classes_num_filtered = np.argmax(probs, axis=3)[filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]

    argsort = np.array(np.argsort(probs_filtered))[::-1]
    boxes_filtered = boxes_filtered[argsort]
    probs_filtered = probs_filtered[argsort]
    classes_num_filtered = classes_num_filtered[argsort]

    for i in range(len(boxes_filtered)):
        if probs_filtered[i] == 0:
            continue
        for j in range(i + 1, len(boxes_filtered)):
            if iou(boxes_filtered[i], boxes_filtered[j]) > iou_threshold:
                probs_filtered[j] = 0.0

    filter_iou = np.array(probs_filtered > 0.0, dtype='bool')
    boxes_filtered = boxes_filtered[filter_iou]
    probs_filtered = probs_filtered[filter_iou]
    classes_num_filtered = classes_num_filtered[filter_iou]

    result = []
    for i in range(len(boxes_filtered)):
        result.append(
            [classes[classes_num_filtered[i]], 
             boxes_filtered[i][0], boxes_filtered[i][1], boxes_filtered[i][2], boxes_filtered[i][3], 
             probs_filtered[i]])
    return result

def iou(box1, box2):
    tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
    lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
    if tb < 0 or lr < 0:
        intersection = 0
    else:
        intersection = tb * lr
    return intersection / (box1[2] * box1[3] + box2[2] * box2[3] - intersection)

def show_results(img, results, img_width, img_height, output_img_file):
    img_cp = img.copy()
    img_cp *= 255
    disp_console = True
    imshow = True
    for i in range(len(results)):
        x = int(results[i][1])
        y = int(results[i][2])
        w = int(results[i][3]) // 2
        h = int(results[i][4]) // 2
        if disp_console:
            print output_img_file
            print '   class : ' + results[i][0] + ' , [x,y,w,h]=[' + str(x) + ',' + str(y) + ',' + str(
                int(results[i][3])) + ',' + str(int(results[i][4])) + '], Confidence = ' + str(results[i][5])
        xmin = x - w
        xmax = x + w
        ymin = y - h
        ymax = y + h
        if xmin < 0:
            xmin = 0
        if ymin < 0:
            ymin = 0
        if xmax > img_width:
            xmax = img_width
        if ymax > img_height:
            ymax = img_height
        if imshow:
            cv2.rectangle(img_cp, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.rectangle(img_cp, (xmin, ymin - 20), (xmax, ymin), (125, 125, 125), -1)
            cv2.putText(img_cp, results[i][0] + ' : %.2f' % results[i][5], (xmin + 5, ymin - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.imwrite(output_img_file, img_cp)
    
    '''
    if imshow:
        cv2.imshow('YOLO detection', img_cp)
        cv2.waitKey(1000)
    '''

def make_detection(img_files):
    input_img_file, output_img_file = img_files[0], img_files[1]
    img = caffe.io.load_image(input_img_file)  # load the image using caffe io
    inputs = img
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    # transformer.set_channel_swap('data', (2,1,0))
    out = net.forward_all(data=np.asarray([transformer.preprocess('data', inputs)]))
    img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    results = interpret_output(out['result'][0], img.shape[1], img.shape[0])    
    show_results(img_cv, results, img.shape[1], img.shape[0], output_img_file)

def get_file_list(find_path, flag_str=[]):     
    file_list = []  
    file_names = os.listdir(find_path)  
    if len(file_names) > 0:  
       for fn in file_names:  
           if len(flag_str) > 0:  
               if is_substring(flag_str,fn):  
                   full_filename = os.path.join(find_path,fn)  
                   file_list.append(full_filename)  
           else:  
               full_filename = os.path.join(find_path,fn)  
               file_list.append(full_filename)  
    if len(file_list) > 0:  
        file_list.sort()    
    return file_list 
    
def is_substring(substr_list, Str):  
    flag = True  
    for substr in substr_list:  
        if not (substr in Str):  
            flag = False  
    return flag

def get_imagefile_list(input_txt):
    input_img_list = []
    with open(input_txt, 'r') as fi:
         input_img_list = [line.strip()for line in fi]
    return input_img_list

def get_args():
    parser = argparse.ArgumentParser(description='output the results of prediction')
    parser.add_argument('-m', dest='model', 
        help='yolo network with format .prototxt', default=None, type=str)
    parser.add_argument('-w', dest='weight',
        help='trained weights with format .caffemodel', default=None, type=str)
    parser.add_argument('-i', dest='inpath',
        help='input path of images', default=None, type=str)
    parser.add_argument('-t', dest='intxt',
        help='train.txt or test.txt', default=None, type=str)
    parser.add_argument('-o', dest='outpath', 
        help='output path of results', default=None, type=str)
    parser.add_argument('-c', dest='cpu_num', 
        help='used cpu numbers for multiprocessing', default=8, type=int)
    parser.add_argument('-p', dest='prob_threshold', 
        help='threshold for class_prob*confidence', default=0.11, type=float)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args    

if __name__ == '__main__':

    caffe.set_mode_cpu()

    args = get_args()

    model_file = args.model
    weight_file = args.weight
    if(args.inpath):
       input_dir = args.inpath
       input_img_list = get_file_list(input_dir, ['.jpg'])
    if(args.intxt):
       input_txt = args.intxt
       input_img_list = get_imagefile_list(input_txt)
    output_dir = args.outpath
    if not os.path.exists(output_dir):
       os.makedirs(output_dir)
    output_img_list = [os.path.join(output_dir, img_file[img_file.rfind('/')+1:]) for img_file in input_img_list]
    cpu_num = args.cpu_num
    global prob_threshold
    prob_threshold = args.prob_threshold
    
    tic = time.clock()
    print "the number of testing samples:"+str(len(output_img_list))
    print "test, begin..."
    time.sleep(5)
    global net
    net = caffe.Net(model_file, weight_file, caffe.TEST)
    pool=Pool(cpu_num)
    pool.map(make_detection, zip(input_img_list, output_img_list))
    toc = time.clock()
    print('running time:{:.3f} seconds'.format(toc - tic))
    
