#*****************************
#date:2016-07-26
#author:luolongqiang
#*****************************
import os, sys, time, argparse
import pandas as pd
from multiprocessing import Pool, cpu_count
from EvalFinal import GetFileList, IsSubString, GetCategories

def FilterWithThreshold(args):
    input_txt, output_txt = args[0], args[1]
    fi = open(input_txt, 'r') 
    fo = open(output_txt, 'w')
    while True:
        s = fi.readline()
        if not s:
            break
        obj = s.strip().split(' ') 
        if '_' in obj[0]:
            name = obj[0].split('_')[0]
        else:
            name = obj[0]
        proba = float(obj[5])
        if name in categories and proba >= thre_dict[name]:
            fo.write(s)
    fi.close()
    fo.close()
   
def GetArgs():
    parser = argparse.ArgumentParser(description='Filter data with optimal threshold')
    parser.add_argument('-cfg', dest='cfg_FilterWithThreshold', \
        help='config file for filtering dataset', default=None, type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

if __name__=='__main__':  
    tic = time.clock()

    global categories, thre_dict

    if '-cfg' in sys.argv:
        args = GetArgs()

        conf = {}
        with open(args.cfg_FilterWithThreshold, 'r') as f:
            for line in f:
                infos = line.strip().split('=')
                conf[infos[0]] = infos[1]

        categories = conf['cls_names'].split(',')    

        input_txt_list = []
        output_txt_list = []
        input_txt_root = conf['input_txt_root']
        output_txt_root = conf['output_txt_root']
        for txt_dir in conf['input_txt_dir'].split(','):
            input_txt_dir = os.path.join(input_txt_root, txt_dir)
            temp_list = GetFileList(input_txt_dir, ['.txt'])
            input_txt_list += temp_list
            output_txt_dir = output_txt_root
            for idir in txt_dir.split('/'):
                output_txt_dir = os.path.join(output_txt_dir, idir)
                if not os.path.exists(output_txt_dir):
                    os.mkdir(output_txt_dir)
            for pth in temp_list:
                complete_path = os.path.join(output_txt_dir, pth.split('/')[-1])
                output_txt_list.append(complete_path) 

        thre_df = pd.read_csv(conf['optimal_threshold_csv_path'])
        thre_dict = {categories[i]:thre_df['optimal_threshold'][i] for i in range(len(categories))}      
    else:
        if len(sys.argv) == 1:
            args = GetArgs()
        
        categories = ['apple','beer','broccoli','chineseCabbage','cucumber','egg',\
                      'eggplant','grape','ham','milk','onion','orange','bitter','pear',\
                      'potato','radish','strawberry','tomato','watermelon','whiteradish']
                      
        if '-c' in sys.argv:
            categories = GetCategories(sys.argv[sys.argv.index('-c')+1]) 

        input_file = GetFileList(sys.argv[1])
        output_file = sys.argv[2]
        if not os.path.exists(output_file):
            os.mkdir(output_file)

        input_txt_list = []
        for pth in input_file:
            input_txt_list += GetFileList(pth, ['.txt'])
            temp = output_file + pth[pth.rfind('/'):]
            if not os.path.exists(temp):
                os.mkdir(temp)

        output_txt_list = []
        for pth in input_txt_list:
            line_list = pth.split('/')
            temp = output_file + '/' + line_list[-2] + '/' + line_list[-1]
            output_txt_list.append(temp)

        thre_df = pd.read_csv(sys.argv[3])
        thre_dict = {categories[i]:thre_df['optimal_threshold'][i] for i in range(len(categories))}


    pool = Pool(int(cpu_count()*7/8)) 
    pool.map(FilterWithThreshold, zip(input_txt_list, output_txt_list))  
    
    toc = time.clock()
    print('running time:{:.3f} seconds'.format(toc-tic))
      
