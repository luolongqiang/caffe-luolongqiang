import os, sys, time, argparse
import numpy as np
from numpy import array

def get_none0_labels(input_txt, output_txt):
	fo = open(output_txt, 'w')
	with open(input_txt, 'r') as fi:
		for line in fi: 
			multilabel = map(int, line.strip().split()[1:])
			if sum(multilabel) != 0:
				fo.write(line)
	fi.close()
	fo.close()

def get_args():
    parser = argparse.ArgumentParser(description='delete samples with all 0-label') 
    parser.add_argument('-i', dest='input_txt',
        help='input_style_multilabel.txt', default=None, type=str)   
    parser.add_argument('-o', dest='output_txt',
        help='output_style_multilabel.txt', default=None, type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = get_args()
    input_txt = args.input_txt
    output_txt = args.output_txt
    
    tic = time.clock()
    get_none0_labels(input_txt, output_txt)
    toc = time.clock()
    print 'running time:{} seconds'.format(toc-tic)