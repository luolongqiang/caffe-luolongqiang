import numpy as np
from numpy import array
from pandas import DataFrame
import os, sys, time, argparse
from multiprocessing import Pool, cpu_count
import cv2, caffe
caffe.set_mode_cpu()
 
# python python/shoes/test.py -m models/shoes_vgg19_bn/deploy.prototxt -w models/shoes_vgg19_bn/vgg19_bn_ft_out/_iter_10000.caffemodel -i data/shoes/shoes_test_label.txt -o models/shoes_vgg19_bn/results/1w.txt -s models/shoes_vgg19_bn/results/vgg19_bn_1w.csv -c 16 -n 10

def make_prediction(img_file_label):
    img_file_name = img_file_label.split()[0]
    real_label = int(img_file_label.split()[-1])

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
    pred_label = np.argmax(probs)
    if real_label!=pred_label:
        print '(TorF, real, pred, prob, img): ({}, {}, {}, {:.3f}, {})'.\
         format(real_label==pred_label, real_label, pred_label, probs[pred_label], img_file_name)
    return real_label, pred_label

def get_confusion_matrix(output_csv, class_num, results):
    conf_mat = np.zeros((class_num, class_num))
    total_num, acc_num = 0, 0
    for real_pred in results:
        real_label, pred_label = real_pred[0], real_pred[1]
        conf_mat[real_label, pred_label] += 1
        if real_label == pred_label:
            acc_num += 1
        total_num += 1
    accuracy = acc_num*1.0 / total_num
    classes =  range(class_num)
    raw_total_num = np.sum(conf_mat,1)
    precisions = [conf_mat[i,i]*1.0/raw_total_num[i] for i in classes]

    df = {cls: conf_mat[:,i] for i, cls in enumerate(classes)}
    df['class'] = classes
    df['row_total_num'] = raw_total_num
    df['precision'] = precisions
    df = DataFrame(df)[['class'] + classes + ['row_total_num', 'precision']]
    if output_csv:
        df.to_csv(output_csv, index = False)

    print 'confusion matrix:\n{}'.format(df)
    print 'accuracy:{}'.format(accuracy)

def output_predict_results(img_list, output_txt, pred_list):
    fo = open(output_txt, 'w')
    for img_label, pred in zip(img_list, pred_list):
        line = img_label + ' ' + str(pred) + '\n'
        fo.write(line)
    fo.close()

def get_args():
    parser = argparse.ArgumentParser(description = 'output the results of prediction')
    parser.add_argument('-m', dest = 'model', 
        help = 'network with format .prototxt', default = None, type = str)
    parser.add_argument('-w', dest = 'weight',
        help = 'trained weights with format .caffemodel', default = None, type = str)
    parser.add_argument('-i', dest = 'in_txt',
        help = 'input txt file including pathes of testing images', default = None, type = str)
    parser.add_argument('-o', dest = 'out_txt', 
        help = 'output path of results', default = None, type = str)
    parser.add_argument('-s', dest = 'out_csv',
        help = 'confusion_matrix.csv', default = None, type = str)
    parser.add_argument('-c', dest = 'cpu_num', 
        help = 'used cpu number for multiprocessing', default = 16, type = int)
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
    input_txt   = args.in_txt
    output_txt  = args.out_txt
    output_csv  = args.out_csv
    cpu_num     = args.cpu_num
    class_num   = args.class_num

    tic  = time.time()

    img_list = open(input_txt, 'r').read().splitlines()
    net  = caffe.Net(model_file, weight_file, caffe.TEST)
    pool = Pool(cpu_num)
    results = pool.map(make_prediction, img_list)
    get_confusion_matrix(output_csv, class_num, results)
   
    if output_txt:
        output_dir = output_txt[:output_txt.rfind('/')]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_predict_results(img_list, output_txt, array(results)[:,1])
     
    toc  = time.time()
    print 'running time:{:.3f} minutes'.format((toc - tic)/60.0)
    
