import numpy as np
from numpy import array
import os, sys, time, argparse
from multiprocessing import Pool, cpu_count
import cv2, caffe
caffe.set_mode_cpu()

# python python/deepFashion/test_multilabel.py -m models/style_resnet50_bn/deploy.prototxt -w models/style_resnet50_bn/resnet50_bn_ft_out/_iter_20000.caffemodel -i data/deepFashion/style_test_multilabel.txt -o models/style_resnet50_bn/results/2w.txt  -c 2 -t 0.0 -k 20 -mode pretest

def make_prediction(img_file_name):
    origin_img = cv2.imread(img_file_name, 1)  # BRG
    width, height = net.blobs['data'].data.shape[2:]
    resize_img = cv2.resize(origin_img/255.0, (width, height), interpolation=cv2.INTER_LINEAR)
    img_data = np.array(resize_img, dtype=np.float32)
    img_data = img_data[:, :, ::-1]
    img_data = img_data.transpose((2, 0, 1))
    out = net.forward(data = np.asarray([img_data]))
    probs = out['prob'][0]
    return probs

def make_test(img_file_label): 
    img_file_name = img_file_label.split()[0]
    probs = make_prediction(img_file_name)

    top_1_label = np.argmax(probs)
    top_1_prob  = probs[top_1_label]
    print 'top1-label, top1-prob:({}, {:.3f})'.format(top_1_label, top_1_prob)

    new_line = img_file_name
    for i in probs:
        new_line += ' ' + str(i)
    new_line += '\n'

    return new_line

def make_pretest(img_file_label):
    img_file_name = img_file_label.split()[0]
    real_labels = array(map(int, img_file_label.strip().split()[1:]))

    probs = make_prediction(img_file_name)

    cond_pred_set = set(np.argsort(probs)[-topk:])
    pred_set = set([ele for ele in cond_pred_set if probs[ele]>threshold])
    if len(pred_set)==0:
        pred_set = set(np.argsort(probs)[-2:])

    real_set = set([i for i, label in enumerate(real_labels) if label==1])
    acc_set = real_set & pred_set
   # total_set = real_set | pred_set

    if len(real_set) == 0:
        recall = 1
    else:
        recall = float(len(acc_set))/len(real_set)
    print len(real_set), len(pred_set), str(recall)[:4], np.sort(probs)[-3:]

def get_args():
    parser = argparse.ArgumentParser(description = 'output the results of prediction')
    # arguments for pretest & test
    parser.add_argument('-m', dest = 'model', 
        help = 'network with format .prototxt', default = None, type = str)
    parser.add_argument('-w', dest = 'weight',
        help = 'trained weights with format .caffemodel', default = None, type = str)
    parser.add_argument('-i', dest = 'input_txt',
        help = 'input txt file including pathes of testing images', default = None, type = str)
    parser.add_argument('-c', dest = 'cpu_num', 
        help = 'used cpu numbers for multiprocessing', default = 8, type = int)
    parser.add_argument('-mode', dest = 'mode',
        help = 'test or pretest', default = 'test', type = str)
    # arguments for test
    parser.add_argument('-o', dest = 'output_txt', 
        help = 'output path of results', default = None, type = str)
    # arguments for pretest
    parser.add_argument('-k', dest = 'topk',
        help = 'topk', default = 5, type = int)
    parser.add_argument('-t', dest = 'threshold',
        help = 'threshold', default = 0.0, type = float)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args    

if __name__ == '__main__':
    

    global net, topk, threshold

    args = get_args()

    model_file  = args.model
    weight_file = args.weight
    input_txt   = args.input_txt
    cpu_num     = args.cpu_num
    mode        = args.mode
    output_txt  = args.output_txt # argument for test
    topk        = args.topk       # argument for pretest
    threshold   = args.threshold  # argument for pretest
    
    tic  = time.time()

    img_list = open(input_txt, 'r').read().splitlines()

    print "test_num:", len(img_list)
    print "test, begin..."

    net  = caffe.Net(model_file, weight_file, caffe.TEST)
    pool = Pool(cpu_num)

    if mode == 'test':
        if output_txt:
            output_dir = output_txt[:output_txt.rfind('/')]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        results = pool.map(make_test, img_list)
        if output_txt:
            open(output_txt, 'w').writelines(results)
    elif mode == 'pretest':
        pool.map(make_pretest, img_list)

    toc  = time.time()
    print 'running time:{:.3f} minutes'.format((toc - tic)/60.0)
    
