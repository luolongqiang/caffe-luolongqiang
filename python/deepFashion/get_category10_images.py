import numpy as np 
from numpy import array
import os, sys, time, argparse, shutil 

def get_category_images(input_txt, output_dir):
    fi = open(input_txt, 'r')
    cls_name_list = ['3-Blouse', '6-Cardigan', '11-Jacket', '16-Sweater', '17-Tank', '18-Tee', '19-Top', '32-Shorts', '33-Skirt', '41-Dress']

    # output_dir = 'data/deepFashion/category_partition'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)    
    for cls_name in cls_name_list:
        cls_output_dir = output_dir + '/' + cls_name
        if not os.path.exists(cls_output_dir):
            os.mkdir(cls_output_dir)

    for i, line in enumerate(list(fi)):
        line_list = line.strip().split()
        img_file_name = line_list[0]
        img_cls = line_list[-1]
        output_file_name = output_dir + '/' + img_cls + '/' + str(i) + '.jpg'
        shutil.copy(img_file_name, output_file_name)
        print i, output_file_name
    #end

def get_args():
    parser = argparse.ArgumentParser(description='get category images') 
    parser.add_argument('-i', dest='input_txt',
        help='train\val\test.txt', default=None, type=str) 
    parser.add_argument('-d', dest='output_dir',
        help='output_partition_dir', default=None, type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = get_args()
    input_txt = args.input_txt
    output_dir = args.output_dir
    
    tic = time.clock()
    print 'get partition images, begin...'
    get_category_images(input_txt, output_dir)
    print 'get partition images, done'
    toc = time.clock()
    print 'running time:{} seconds'.format(toc-tic)
