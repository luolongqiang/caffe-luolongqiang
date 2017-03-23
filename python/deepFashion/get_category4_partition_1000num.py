import os, sys, argparse, random
# python python/deepFashion/get_4category_partition_1000num.py -i data/deepFashion/category4_train_newlabel.txt -o data/deepFashion/category4_train_1000newlabel.txt -n 1000
def get_4category_partition_1000num(input_txt, output_txt, number):  
    new_line_dict = {'0':[],'1':[],'2':[],'3':[]}
    fi = open(input_txt, 'r')
    for line in fi:
        line_list = line.strip().split()
        file_name = line_list[0]
        label = line_list[-1]
        new_line_dict[label].append(line) 
    fi.close()

    new_line_list = []
    for label in new_line_dict.keys():
        random.shuffle(new_line_dict[label])
        new_line_list += new_line_dict[label][:number]
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
    parser.add_argument('-n', dest='num',
        help='number', default=None, type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = get_args()
    input_txt = args.input_txt
    output_txt = args.output_txt
    number = args.num
    get_4category_partition_1000num(input_txt, output_txt, number)
