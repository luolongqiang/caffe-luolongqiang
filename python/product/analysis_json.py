import os, sys, argparse, time, json, random
from numpy import array
from collections import Counter
from pandas import DataFrame

# python python/product/analysis_json.py -j data/product/product_list.json -c data/product/color_count.csv -k color -r data/product/img -p data/product

def analysis_json(json_file, csv_file, keyword, img_root, partition_dir):
    key_list = []
    with open(json_file, 'r') as fi:
        for line in fi:
            json_dict = json.loads(line.strip())
            for ele in json_dict.values():
                key_list.append(ele[keyword])

    key_num_dict = dict(Counter(key_list))
    key_set = key_num_dict.keys()
    num_set = key_num_dict.values()
    if csv_file:
        df = DataFrame({keyword:key_set, 'num':num_set})
        df = df[[keyword, 'num']]
        df.to_csv(csv_file, index=False, encoding='utf-8')

    key_imglist_dict = {clas:[] for clas in key_set}
    key_label_dict = {clas:str(i) for i, clas in enumerate(key_set)}
    with open(json_file, 'r') as fi:
        for line in fi:
            json_dict = json.loads(line.strip())
            for pth in json_dict.keys():
                clas = json_dict[pth][keyword]
                new_line = img_root + '/' + pth + '.jpg ' + key_label_dict[clas] + '\n'
                key_imglist_dict[clas].append(new_line)
    
    fo_train = open(partition_dir + '/' + keyword + '_train_label.txt', 'w')
    fo_val   = open(partition_dir + '/' + keyword + '_val_label.txt',   'w')
    fo_test  = open(partition_dir + '/' + keyword + '_test_label.txt',  'w')
    for clas in key_set:
        random.shuffle(key_imglist_dict[clas])
        total_num = key_num_dict[clas]
        if total_num == 1:
            train_num, val_num, test_num == 1, 0, 0
        elif total_num == 3:
            train_num, val_num, test_num == 1, 1, 1
        elif total_num == 4:
            train_num, val_num, test_num == 2, 1, 1
        elif total_num == 5:
            train_num, val_num, test_num == 3, 1, 1
        else:
            train_num = int(total_num*0.8)
            val_num = int(round(total_num*0.1))
            test_num = total_num - train_num - val_num
        train_lines = array(key_imglist_dict[clas])[:train_num]
        val_lines   = array(key_imglist_dict[clas])[train_num:train_num+val_num]
        test_lines  = array(key_imglist_dict[clas])[train_num+val_num:]
        fo_train.writelines(train_lines)
        fo_val.writelines(val_lines)
        fo_test.writelines(test_lines) 
    fo_train.close()
    fo_val.close()
    fo_test.close()
    

def get_args():
    parser = argparse.ArgumentParser(description='analysis json file')
    parser.add_argument('-j', dest='json_file',
        help='product.json', default=None, type=str)
    parser.add_argument('-c', dest='csv_file',
        help='category_count.csv', default=None, type=str) 
    parser.add_argument('-k', dest='keyword',
        help='keyword: e.g, category or style', default=None, type=str) 
    parser.add_argument('-r', dest='img_root',
        help='image root', default=None, type=str)   
    parser.add_argument('-p', dest='partition_dir',
        help='partition_dir', default=None, type=str)   
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

if __name__=='__main__':

    tic = time.time()

    args = get_args()

    json_file     = args.json_file
    csv_file      = args.csv_file
    keyword       = args.keyword
    img_root      = args.img_root
    partition_dir = args.partition_dir
    
    analysis_json(json_file, csv_file, keyword, img_root, partition_dir)
    
    toc = time.time()
    print 'running time:{}'.format(toc-tic)