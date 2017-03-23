import os, sys, argparse, random

def get_4category_partition(input_txt, output_txt): 
    label_dict = {'6':'0', '18':'1', '32':'2', '41':'3'} 
    fi = open(input_txt, 'r')
    new_line_list = []
    for line in fi:
        line_list = line.strip().split()
        origin_label = line_list[-1]
        if origin_label in label_dict.keys():
            new_label = label_dict[origin_label]
            file_name = line_list[0]
            new_line_list.append(file_name + ' ' + new_label + '\n') 
    fi.close()

    random.shuffle(new_line_list)
    fo = open(output_txt, 'w')
    fo.writelines(new_line_list)
    fo.close()

def get_args():
    parser = argparse.ArgumentParser(description='transform category label') 
    parser.add_argument('-i', dest='input_txt',
        help='input_txt', default=None, type=str) 
    parser.add_argument('-o', dest='output_txt',
        help='output_txt', default=None, type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = get_args()
    input_txt = args.input_txt
    output_txt = args.output_txt
    get_4category_partition(input_txt, output_txt)
