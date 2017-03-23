import numpy as np
from pandas import DataFrame
import os, sys, time, argparse
# python python/deepFashion/evaluation.py -i models/category4_lenet/results/predicted_labels.txt -o models/category4_lenet/results/confusion_matrix.csv

def get_confusion_matrix(input_txt, class_num):
    conf_mat = np.zeros((class_num, class_num))
    total_num, acc_num = 0, 0
    for line in open(input_txt, 'r'):
        line_list = line.strip().split()
        real_label = int(line_list[-2])
        pred_label = int(line_list[-1])
        conf_mat[real_label, pred_label] += 1
        if real_label == pred_label:
            acc_num += 1
        total_num += 1
    accuracy = acc_num*1.0 / total_num
    return conf_mat, accuracy

def get_args():
    parser = argparse.ArgumentParser(description='evaluation the results of prediction')
    parser.add_argument('-i', dest='input_txt',
        help='real_pred_labels.txt', default=None, type=str)
    parser.add_argument('-o', dest='output_csv',
        help='confusion_matrix.csv', default=None, type=str)
    parser.add_argument('-n', dest='class_num',
        help='class number', default=4, type=int)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

if __name__=='__main__':
   
   args = get_args()

   input_txt  = args.input_txt
   output_csv = args.output_csv
   class_num  = args.class_num
   classes = range(class_num)

   tic = time.time()

   conf_mat, accuracy = get_confusion_matrix(input_txt, class_num) 
   total_num = np.sum(conf_mat,1)
   precisions = [conf_mat[i,i]*1.0/total_num[i] for i in classes]
   
   df = {cls: conf_mat[:,i] for i, cls in enumerate(classes)}
   df['class'] = classes
   df['total_num'] = total_num
   df['precision'] = precisions
   df = DataFrame(df)[['class'] + classes + ['total_num', 'precision']]
   df.to_csv(output_csv, index = False)

   print 'confusion matrix:\n{}'.format(df)
   print 'accuracy:{}'.format(accuracy)
      
   toc = time.time()
   print 'running time:{} seconds'.format(toc-tic)

