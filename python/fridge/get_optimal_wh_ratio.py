#********************
#date:2016-08-24
#author:luolongqiang
#********************

import os, sys, time, argparse, itertools, math, copy
import numpy as np
from numpy import array
from pandas import DataFrame
from multiprocessing import Pool, cpu_count
from EvalFinal import IsSubString, GetFileList

def get_labels(input_txt):
    labels = []
    with open(input_txt, 'r') as fi:
        for line in fi:
            name = line.split(' ')[0]
            if 'hard' in name:
                name = 'hard'
            else:
                if '_' in name:
                    name = name.split('_')[0]
            labels.append(name)
    fi.close()
    return labels

def get_label_dict(args):
    input_txt,categories,threshold = args[0], args[1],args[2]
    label_dict = {ele:array(5*[0]) for ele in categories}
    with open(input_txt, 'r') as fi:
        for line in fi:
            obj = line.strip().split(' ')
            name = obj[0]
            if 'hard' in name:
                name = 'hard'
            else:
                if '_' in name:
                    name = name.split('_')[0]
            wh_ratio = float(obj[3])/float(obj[4])
            if wh_ratio > 1/threshold:
                label_dict[name][1] += 1
            elif wh_ratio < threshold:
                label_dict[name][2] += 1
            else:
                label_dict[name][3] += 1
    fi.close()
    return label_dict

def get_entropy(vec):
    s = sum(vec)
    if s > 0:
        p_vec = vec*1.0/sum(vec)
    else:
        p_vec = vec
    entropy = 0
    for p in p_vec:
        if p>0:
           entropy += -p*math.log(p)
    return entropy

def get_args():
    parser = argparse.ArgumentParser(description='the statistics of w-h ratio for each category')
    parser.add_argument('-in', dest='input_root', 
        help='input-root of real labels', default=None, type=str)
    parser.add_argument('-out', dest='output_dir', 
        help='output-dir of optimal w-h ratio for each category', default=None, type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args 

if __name__=='__main__':
  
    tic = time.clock()
    global categories, threshold, vec_num

    args = get_args()
    input_root = args.input_root
    output_dir = args.output_dir
    
    input_txt_list = []
    for input_dir in os.listdir(input_root):
        input_txt_dir = os.path.join(input_root, input_dir, 'txt')
        input_txt_list += GetFileList(input_txt_dir, ['.txt'])
    num = len(input_txt_list)

    pool = Pool(int(cpu_count()*7/8))
    labels = pool.map(get_labels, input_txt_list)
    categories = list(set(list(itertools.chain.from_iterable(labels))))
    
    optimal_infs = {ele:5*[0] for ele in categories}
    for threshold in np.arange(0.1, 1.0, 0.1):
        results = pool.map(get_label_dict, zip(input_txt_list, [categories]*num, [threshold]*num)) 
        label_infs = {}
        for ele in categories:
            label_infs[ele] = sum([label_dict[ele] for label_dict in results])
            entropy = get_entropy(label_infs[ele][1:-1])
            label_infs[ele] = [threshold] + list(label_infs[ele][1:-1]) + [entropy]
            if optimal_infs[ele][-1] < entropy:
                optimal_infs[ele] = label_infs[ele]

    df = {'categories':categories}
    df_header = ['wh_ratio','heng_num','shu_num','xie_num','entropy']
    for i in range(len(df_header)):
        df[df_header[i]] = [optimal_infs[ele][i] for ele in categories]
    df = DataFrame(df)
    df = df[['categories'] + df_header]
    df.to_csv(output_dir + '/optimal_wh_ratio.csv', index = False)

    toc = time.clock()
    print('running time:{:.3f} seconds'.format(toc-tic))



   
