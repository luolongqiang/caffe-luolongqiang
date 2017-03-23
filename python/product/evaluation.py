import numpy as np
from numpy import array
from pandas import DataFrame
import os, sys, time, argparse
# python python/product/evaluation.py -r data/product/multi_task_test_label.txt -p models/multi_task_vgg16_bn/results/8w.txt -o 1

###############################################################################80

def evaluation(real_txt, pred_txt, is_out):
    real_line_list = open(real_txt, 'r').read().splitlines()
    pred_line_list = open(pred_txt, 'r').read().splitlines()
    real_mat, pred_mat = [], []
    for real_line, pred_line in zip(real_line_list, pred_line_list):
        real_labels = map(int, real_line.strip().split()[1:])
        pred_labels = map(int, pred_line.strip().split()[1:])
        real_mat.append(real_labels)
        pred_mat.append(pred_labels)
    real_mat,  pred_mat = array(real_mat), array(pred_mat)

    sty_inx = array(real_mat[:, 1] != 0)
    col_inx = real_mat[:, 2] != 0
    tex_inx = real_mat[:, 3] != 0
    category_acc = sum(real_mat[:, 0]==pred_mat[:, 0])/float(len(real_line_list))
    style_acc    = sum(real_mat[sty_inx, 1]==pred_mat[sty_inx, 1])/float(sum(sty_inx))
    color_acc    = sum(real_mat[col_inx, 2]==pred_mat[col_inx, 2])/float(sum(col_inx))
    texture_acc  = sum(real_mat[tex_inx, 3]==pred_mat[tex_inx, 3])/float(sum(tex_inx))

    print 'category_acc:{:.3f}, style_acc:{:.3f}, color_acc:{:.3f}, texture_acc:{:.3f}'.\
        format(category_acc, style_acc, color_acc, texture_acc)

    if is_out:
        output_dir = pred_txt[:pred_txt.rfind('/')]
        iter_ = pred_txt.split('/')[-1].split('.')[0]
        get_conf_mat(real_mat[:, 0], pred_mat[:, 0], 'category', 80, iter_, output_dir)
        get_conf_mat(real_mat[:, 1], pred_mat[:, 1], 'style', 56, iter_, output_dir)
        get_conf_mat(real_mat[:, 2], pred_mat[:, 2], 'color', 24, iter_, output_dir)
        get_conf_mat(real_mat[:, 3], pred_mat[:, 3], 'texture', 47, iter_, output_dir)
    #end

def get_conf_mat(real_labels, pred_labels, task, class_num, iter_, output_dir):
    conf_mat = np.zeros((class_num, class_num))
    if task=='category':
        for real, pred in zip(real_labels, pred_labels):
            conf_mat[real, pred] += 1
        classes = range(0, class_num)
        total_num = np.sum(conf_mat, 1)
        precisions = [conf_mat[i, i]*1.0/total_num[i] for i in classes]
        df = {i: conf_mat[:, i] for i in classes}
    elif task=='style' or task=='color' or task=='texture':
        for real, pred in zip(real_labels, pred_labels):
            if real != 0:
                conf_mat[real, pred] += 1
        classes = range(1, class_num)
        total_num = np.sum(conf_mat[1:, 1:],1)
        precisions = [conf_mat[i, i]*1.0/total_num[i-1] for i in classes]
        df = {i: conf_mat[1:, i] for i in classes}   
    else:
        sys.exit(1)

    df['class'] = classes
    df['total_num'] = total_num
    df['precision'] = precisions
    df = DataFrame(df)[['class'] + classes + ['total_num', 'precision']]
    df.to_csv(output_dir + '/' + task + '_' + iter_ + '.csv', index = False)

def get_args():
    parser = argparse.ArgumentParser(description='evaluation the results of prediction')
    parser.add_argument('-r', dest='real_txt',
        help='real_labels.txt', default=None, type=str)
    parser.add_argument('-p', dest='pred_txt',
        help='pred_labels.txt', default=None, type=str)
    parser.add_argument('-o', dest='is_out',
        help='output_dir of results', default=1, type=int)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

if __name__=='__main__':
   
   args = get_args()

   real_txt = args.real_txt
   pred_txt = args.pred_txt
   is_out   = args.is_out

   print '------------------------'*3
   tic = time.time()
   evaluation(real_txt, pred_txt, is_out)
   toc = time.time()
   print '------------------------'*3

   print 'running time:{} seconds'.format(toc-tic)

