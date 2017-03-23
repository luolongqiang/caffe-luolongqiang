import numpy as np
from numpy import array
from pandas import DataFrame
import matplotlib.pyplot as plt
from multiprocessing import Pool
import os, sys, time, argparse

# python python/deepFashion/evaluation_multilabel.py -r data/deepFashion/style_test_multilabel.txt -p models/style_alexnet/results/imgnet_ft_bn_140w.txt -c 16 -t 0.0 -k 20

def get_real_and_pred_mat(real_txt, pred_txt):
    fo_real = open(real_txt, 'r')
    fo_pred = open(pred_txt, 'r')
    real_mat, pred_mat = [], []
    for real_line, pred_line in zip(fo_real, fo_pred):
        real_labels = map(int, real_line.strip().split()[1:])
        pred_labels = get_pred_labels(real_line, pred_line)
        real_mat.append(real_labels)
        pred_mat.append(pred_labels)
    real_mat = array(real_mat)
    pred_mat = array(pred_mat)
    return real_mat, pred_mat 

def get_pred_labels(real_line, pred_line):
    global topk, threshold
    probs = map(float, pred_line.strip().split()[1:])
    cond_pred_set = set(np.argsort(probs)[-topk:])
    pred_set = set([ele for ele in cond_pred_set if probs[ele]>threshold])
    if len(pred_set)==0:
        pred_set = set(np.argsort(probs)[-2:])

    pred_labels = np.zeros(len(probs))
    for i in pred_set:
        pred_labels[i] = 1 

    return pred_labels

def evaluation(real_pred):
    real_list, pred_list = real_pred[0], real_pred[1]
    real_set = set([i for i, ele in enumerate(real_list) if ele==1])
    pred_set = set([i for i, ele in enumerate(pred_list) if ele==1])
    inter_set = real_set & pred_set
    union_set = real_set | pred_set
    
    if len(union_set)==0:
        accuracy = 1
        precision = 1
        recall = 1
        F1_score = 1
    else:
        inter_num = float(len(inter_set))
        accuracy = inter_num/len(union_set)
        if len(pred_set)==0:
            precision = 0
            recall = inter_num/len(real_set)
            F1_score = 0
        elif len(real_set)==0:
            precision = inter_num/len(pred_set)
            recall = 0
            F1_score = 0
        else:
            precision = inter_num/len(pred_set)
            recall = inter_num/len(real_set)
            if precision+recall==0:
                F1_score = 0
            else:
                F1_score = 2*precision*recall/(precision+recall)
    return [accuracy, precision, recall, F1_score]

def output_results(recalls, output_csv):
    labels = range(len(recalls))
    df = DataFrame({'label':labels, 'recall':recalls})
    df = df[['label', 'recall']]
    df.to_csv(output_csv,  index = False)

    plt.bar(labels, recalls, color = "blue")
    plt.xlabel("label")
    plt.ylabel("recall")
    plt.title("label vs recall")
    plt.xlim(0, max(labels) + 1)
    plt.ylim(0, max(recalls) + 0.1)
    plt.savefig(output_csv.replace('csv','jpg'), dpi=300)
    plt.close()

def get_args():
    parser = argparse.ArgumentParser(description='evaluation the results of prediction')
    parser.add_argument('-r', dest='real_txt',
        help='test_labels.txt', default=None, type=str)
    parser.add_argument('-p', dest='pred_txt',
        help='predicted_labels.txt', default=None, type=str)  
    parser.add_argument('-e', dest='exam_res_csv',
        help='example_results.csv', default=None, type=str)
    parser.add_argument('-l', dest='label_res_csv',
        help='label_results.csv', default=None, type=str)
    parser.add_argument('-k', dest = 'topk',
        help = 'topk', default = 5, type = int)
    parser.add_argument('-t', dest = 'threshold',
        help = 'threshold', default = 0.0, type = float)
    parser.add_argument('-c', dest='cpu_num',
        help='cpu number', default=8, type=int)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

if __name__=='__main__':
   
   args = get_args()

   real_txt      = args.real_txt
   pred_txt      = args.pred_txt
   exam_res_csv  = args.exam_res_csv
   label_res_csv = args.label_res_csv
   topk          = args.topk       
   threshold     = args.threshold  
   cpu_num       = args.cpu_num

   tic = time.time()

   real_mat, pred_mat = get_real_and_pred_mat(real_txt, pred_txt)
   pool = Pool(cpu_num)

   exam_res = pool.map(evaluation, zip(real_mat, pred_mat))
   a = np.mean(exam_res, 0)
   print '---based example results---'
   print 'accuarcy:{}, precision:{}, recall:{}, F1_score:{}'.format(a[0],a[1],a[2],a[3])
   
   label_res = pool.map(evaluation, zip(real_mat.T, pred_mat.T))
   b = np.mean(label_res,0)
   print '--- based label results ---'
   print 'accuarcy:{}, precision:{}, recall:{}, F1_score:{}'.format(b[0],b[1],b[2],b[3])

   if label_res_csv:
       output_results(array(label_res)[:,2], label_res_csv)

   toc  = time.time()
   print 'running time:{:.3f} seconds'.format(toc - tic)

