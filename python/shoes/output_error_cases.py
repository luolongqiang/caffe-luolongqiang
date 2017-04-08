import pandas as pd
import shutil
import argparse
import sys, os

def get_args():
    parser = argparse.ArgumentParser(description='output the error cases')
    parser.add_argument('-c', dest='in_csv',
        help='_eval_results.csv', default=None, type=str)
    parser.add_argument('-i', dest='in_dir',
        help='input directory including all images', default=None, type=str)
    parser.add_argument('-o', dest='out_dir',
        help='output directory including error images', default=None, type=str)
    parser.add_argument('-t', dest='threshold', 
        help='the iou threshold for output', default=0.0, type=float)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args
if __name__=='__main__':

    args = get_args()

    eval_results_csv = args.in_csv
    input_dir = args.in_dir
    out_dir = args.out_dir
    threshold = args.threshold

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    df = pd.read_csv(eval_results_csv)
    error_cases = df['image'][df['IoU']<=threshold]
    for ele in error_cases:
        shutil.copy(input_dir+'/'+ele, out_dir+'/'+ele)    
