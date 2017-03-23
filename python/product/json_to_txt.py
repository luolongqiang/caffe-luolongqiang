#coding=utf-8
import os, sys, argparse, time, json, random, codecs
from numpy import array
from collections import Counter
from pandas import DataFrame

# python python/product/json_to_txt.py -j data/product/product_list.json -t data/product/product_list.txt

def json_to_txt(json_file, txt_file):  
    fi = codecs.open(json_file, 'r')
    fo = codecs.open(txt_file, 'w', encoding='utf-8')
    for line in fi:
        json_dict = json.loads(line.strip())
        for pth in json_dict.keys():
            obj = json_dict[pth]
            category = str(obj['category'])
            style = str(obj['style'])
            color = str(obj['color'])
            texture = str(obj['texture'])
            new_line = 'data/product/img/' + pth + '.jpg ' + ' ' + category + ' ' + style + ' ' + color + ' ' + texture + '\n'
            fo.write(new_line)
    fi.close()
    fo.close()

def get_args():
    parser = argparse.ArgumentParser(description='analysis json file')
    parser.add_argument('-j', dest='json_file',
        help='product_list.json', default=None, type=str)
    parser.add_argument('-t', dest='txt_file',
        help='product_list.txt', default=None, type=str)   
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

if __name__=='__main__':

    tic = time.time()

    args = get_args()
    json_file = args.json_file
    txt_file  = args.txt_file
    json_to_txt(json_file, txt_file)
    
    toc = time.time()
    print 'running time:{}'.format(toc-tic)