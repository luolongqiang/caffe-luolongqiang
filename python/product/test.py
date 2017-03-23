import numpy as np
from numpy import array
import os, sys, time, argparse
from multiprocessing import Pool, cpu_count
import cv2, caffe
caffe.set_mode_cpu()

# python python/product/test.py -m models/multi_task_vgg19_bn/deploy.prototxt -w models/multi_task_vgg19_bn/vgg19_bn_ft_out/_iter_20000.caffemodel -i data/product/multi_task_test_label.txt -o models/multi_task_vgg19_bn/results/2w.txt  -c 16

def make_prediction(img_file_label):
    img_file_name = img_file_label.split()[0]
    labels = map(int, img_file_label.split()[1:])

    origin_img = cv2.imread(img_file_name, 1)  # BRG
    width, height = net.blobs['data'].data.shape[2:]
    resize_img = \
       cv2.resize(origin_img/256.0, (width, height), interpolation=cv2.INTER_LINEAR)
    img_data = np.array(resize_img, dtype=np.float32)
    img_data = img_data[:, :, ::-1]
    img_data = img_data.transpose((2, 0, 1))
    out = net.forward(data = np.asarray([img_data]))
    category_label = np.argmax(out['category_prob'][0])
    style_label = get_predicted_label(out['style_prob'][0])
    color_label = get_predicted_label(out['color_prob'][0])
    texture_label = get_predicted_label(out['texture_prob'][0])

    print 'category, style, color, texture: ({}:{}, {}:{}, {}:{}, {}:{})'.\
       format(labels[0], category_label, labels[1], style_label, \
          labels[2], color_label, labels[3], texture_label)

    new_line = img_file_name + ' ' + str(category_label) + ' ' \
        + str(style_label) + ' ' + str(color_label) + ' ' + str(texture_label) + '\n'

    return new_line

def get_predicted_label(probs):
    max_two = np.argsort(probs)[-2:]
    if max_two[1] == 0:
        label = max_two[0]
    else:
        label = max_two[1]
    return label

def get_args():
    parser = argparse.ArgumentParser(description = 'output the results of prediction')
    parser.add_argument('-m', dest = 'model', 
        help = 'network with format .prototxt', default = None, type = str)
    parser.add_argument('-w', dest = 'weight',
        help = 'trained weights with format .caffemodel', default = None, type = str)
    parser.add_argument('-i', dest = 'input_txt',
        help = '.txt including testing images', default = None, type = str)
    parser.add_argument('-o', dest = 'output_txt', 
        help = 'output path of results', default = None, type = str)
    parser.add_argument('-c', dest = 'cpu_num', 
        help = 'used cpu numbers for multiprocessing', default = 8, type = int)

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
    input_txt   = args.input_txt
    output_txt  = args.output_txt
    cpu_num     = args.cpu_num
    
    tic  = time.time()

    img_list = open(input_txt, 'r').read().splitlines()

    print "test_num:", len(img_list)
    print "test, begin..."

    net  = caffe.Net(model_file, weight_file, caffe.TEST)
    pool = Pool(cpu_num)
    results = pool.map(make_prediction, img_list)

    if output_txt:
        output_dir = output_txt[:output_txt.rfind('/')]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        open(output_txt, 'w').writelines(results)
    
    toc  = time.time()
    print 'running time:{:.3f} minutes'.format((toc - tic)/60.0)
    
