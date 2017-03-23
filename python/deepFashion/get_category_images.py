import numpy as np 
from numpy import array
import os, sys, time, argparse, shutil 

# python python/deepFashion/get_category_images.py -i data/deepFashion/category_test_label.txt -n data/deepFashion/Anno/list_category_cloth.txt -m data/deepFashion/Anno/list_category_img.txt -o data/deepFashion/category_test_imgs

def mkdir_for_partition_category(input_names, input_imgs, output_dir):
    cls_name_list = []
    with open(input_names, 'r') as fi:
        for line in list(fi)[2:]:
            cls_name_list.append(line.split()[0])
    fi.close()

    label_list = []
    with open(input_imgs, 'r') as fi:
        for line in list(fi)[2:]:
            label_list.append(line.strip().split()[-1])
    fi.close()
    labels = np.sort(np.array(list(set(label_list)), dtype=np.int))
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    output_dir_dict = {}
    for i, label in enumerate(labels):
        output_dir_name = output_dir + '/' + str(i) + '-' +str(label) + '-' + cls_name_list[label-1]
        output_dir_dict[str(i)] = output_dir_name
        print output_dir_name
        if not os.path.exists(output_dir_name):
            os.mkdir(output_dir_name)
    return output_dir_dict

def get_category_images(input_txt, input_names, input_imgs, output_dir):
    output_dir_dict = mkdir_for_partition_category(input_names, input_imgs, output_dir)
    with open(input_txt, 'r') as fi:
        for i, line in enumerate(list(fi)):
            line_list = line.strip().split()
            img_file_name = line_list[0]
            new_label = line_list[-1]
            output_file_name = output_dir_dict[new_label] + '/' + str(i) + '.jpg'
            shutil.copy(img_file_name, output_file_name)
            print output_file_name
    #end

def get_args():
    parser = argparse.ArgumentParser(description='get category images') 
    parser.add_argument('-i', dest='input_txt',
        help='train\val\test.txt', default=None, type=str)
    parser.add_argument('-n', dest='input_names',
        help='list_category_cloth.txt', default=None, type=str)
    parser.add_argument('-m', dest='input_imgs',
        help='list_category_img.txt', default=None, type=str)
    parser.add_argument('-o', dest='output_dir',
        help='output_partition_dir', default=None, type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = get_args()
    input_txt = args.input_txt
    input_names = args.input_names
    input_imgs = args.input_imgs
    output_dir = args.output_dir
    
    tic = time.clock()
    get_category_images(input_txt, input_names, input_imgs, output_dir)
    toc = time.clock()
    print 'running time:{} seconds'.format(toc-tic)
