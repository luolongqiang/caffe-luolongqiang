import numpy as np
from numpy import array
import os, sys, time, argparse
from multiprocessing import Pool, cpu_count
import cv2, caffe
caffe.set_mode_cpu()
 
def make_prediction(img_file_label):
    img_file_name = img_file_label.split()[0]
    real_label = int(img_file_label.split()[-1])

    img = caffe.io.load_image(img_file_name)  # load the image using caffe io
    inputs = cv2.resize(img, (resize_num, resize_num), interpolation=cv2.INTER_LINEAR)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    mu = array([182, 186, 186])
    transformer.set_mean('data', mu)
    transformer.set_raw_scale('data', 256) 
    transformer.set_channel_swap('data', (2,1,0))

    out = net.forward(data = np.asarray([transformer.preprocess('data', inputs)]))
    probs = out['prob'][0]
    predicted_label = np.argmax(probs)
    new_line = img_file_label.strip() + ' ' +  str(predicted_label) + '\n'
    print real_label == predicted_label
    return new_line

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
    parser.add_argument('-c', dest = 'cpu_num', 
        help = 'used cpu numbers for multiprocessing', default = 6, type = int)
    parser.add_argument('-r', dest = 'resize_num',
        help = 'resize', default = 256, type = int)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args    

if __name__ == '__main__':

    global net, resize_num  

    args = get_args()

    model_file     = args.model
    weight_file    = args.weight
    input_txt      = args.in_txt
    output_txt     = args.out_txt
    cpu_num        = args.cpu_num
    resize_num     = args.resize_num

    output_dir = output_txt[:output_txt.rfind('/')]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tic  = time.time()

    img_list = open(input_txt, 'r').read().splitlines()
    test_num = len(img_list)

    print "the number of testing samples:"+str(test_num)
    print "test, begin..."

    net  = caffe.Net(model_file, weight_file, caffe.TEST)
    pool = Pool(cpu_num)
    results = pool.map(make_prediction, img_list)
    open(output_txt, 'w').writelines(results)
    #accuracy = sum(array(results)[:,1])*1.0 / test_num 
    #print "accuracy:{}".format(accuracy)
     
    toc  = time.time()
    print 'running time:{:.3f} minutes'.format((toc - tic)/60.0)
    
