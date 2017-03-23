import os, sys, argparse

def transform_label(input_txt, output_txt): 
    label_list = ['3', '6', '11', '16', '17', '18', '19', '32', '33', '41']
    label_dict = {label:str(i) for i, label in enumerate(label_list)} 
    fi = open(input_txt, 'r')
    fo = open(output_txt, 'w')
    for line in fi:
        origin_label = line.strip().split()[-1] 
        new_label = label_dict[origin_label]
        file_name = line.strip().split()[0]
        fo.write(file_name + ' ' + new_label + '\n') 
    fi.close()
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
    transform_label(input_txt, output_txt)
