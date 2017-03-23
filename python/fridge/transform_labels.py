#***********************************
#date:2016-07-25
#author:luolongqiang
#***********************************

import os, sys, time, argparse
from multiprocessing import Pool, cpu_count
from EvalFinal import GetFileList, IsSubString

def modify_labels_to_hsx(args):
    global threshold
    input_txt, output_txt = args[0], args[1]
    fi = open(input_txt,'r')
    fo = open(output_txt,'w')
    while True:
         s = fi.readline()
         if not s:
             break
         obj = s.strip().split(' ')
         name = obj[0]
         if '_' in obj[0]:        
             name = obj[0].split('_')[0] 
         if name in handle_categories1:
             wh_ratio= float(obj[3])/float(obj[4])
             if wh_ratio > 1/threshold:
                 fo.write('H' + s)
             elif wh_ratio < threshold:
                 fo.write('S' + s)
             else:
                 fo.write('X' + s)
         else:
             fo.write(s)
    fi.close()
    fo.close()

def modify_labels_to_origin(args):
    input_txt, output_txt = args[0], args[1]
    fi = open(input_txt,'r')
    fo = open(output_txt,'w')
    while True:
         s = fi.readline()
         if not s:
             break
         obj = s.strip().split(' ')
         name = obj[0]
         if '_' in obj[0]:        
             name = obj[0].split('_')[0]    
         if name in handle_categories2:
             fo.write(s[1:])
         else:
             fo.write(s)
    fi.close()
    fo.close()

def GetArgs():
    parser = argparse.ArgumentParser(description='Transform labels to original labels')
    parser.add_argument('-mode', dest='mode', 
        help='transform mode:to_hsx or to_origin', default='to_hsx', type=str)
    parser.add_argument('-in', dest='input', 
        help='input file path', default=None, type=str)
    parser.add_argument('-out', dest='output', 
        help='output file path', default=None, type=str)
    parser.add_argument('-t', dest='threshold', 
        help='w/h threshold', default=0.7, type=float)    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

if __name__=='__main__':
    tic = time.clock()

    global handle_categories1, handle_categories2, threshold
    handle_categories1 = ['cucumber','eggplant','ham']
    handle_categories2 = ['Hcucumber','Scucumber','Xcucumber',\
                         'Heggplant','Seggplant','Xeggplant',\
                         'Hham','Sham','Xham']
    args = GetArgs()    
    input_txt_root, output_txt_root = args.input, args.output
    input_txt_list, output_txt_list = [], []

    pool = Pool(int(cpu_count()*7/8))
    if args.mode == 'to_hsx':
        threshold = args.threshold
        for input_txt_dir in GetFileList(input_txt_root, []):
            temp_list = GetFileList(input_txt_dir+'/txt', ['.txt'])
            input_txt_list += temp_list
            output_txt_dir = output_txt_root
            for idir in [input_txt_dir.split('/')[-1], 'txt']:
                output_txt_dir = os.path.join(output_txt_dir, idir)
                if not os.path.exists(output_txt_dir):
                    os.mkdir(output_txt_dir)
            for pth in temp_list:
                complete_path = os.path.join(output_txt_dir, pth.split('/')[-1])
                output_txt_list.append(complete_path) 
        pool.map(modify_labels_to_hsx, zip(input_txt_list, output_txt_list))   

    elif args.mode == 'to_origin':
        for input_txt_dir in GetFileList(input_txt_root, []):
            temp_list = GetFileList(input_txt_dir, ['.txt'])
            input_txt_list += temp_list
            output_txt_dir = os.path.join(output_txt_root, input_txt_dir.split('/')[-1])
            if not os.path.exists(output_txt_dir):
                os.mkdir(output_txt_dir)
            for pth in temp_list:
                complete_path = os.path.join(output_txt_dir, pth.split('/')[-1])
                output_txt_list.append(complete_path) 
        pool.map(modify_labels_to_origin, zip(input_txt_list, output_txt_list))

    toc = time.clock()
    print('running time:{:.3f} seconds'.format(toc-tic))

