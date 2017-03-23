#*************************
#date:2016-08-22
#author:luolongqiang
#*************************

import os, sys, time, shutil, argparse
from EvalFinal import GetFileList
from multiprocessing import Pool, cpu_count

def get_pic_of_cluster(cluster_txt):
    global cluster_jpg_root, cat_jpg_dir
    cluster_jpg_dir = os.path.join(cluster_jpg_root, cluster_txt.split('/')[-1].split('.')[0]+'_pic')
    if not os.path.exists(cluster_jpg_dir):
        os.mkdir(cluster_jpg_dir)
    with open(cluster_txt, 'r') as f:
        for s in f:
            line = s.split(' ')
            temp = line[0].split('/')
            jpg_name = temp[0]+'-'+temp[1].split('.')[0] + '-' + line[1] + '.jpg'
            cat_jpg = os.path.join(cat_jpg_dir, jpg_name)
            cluster_jpg = os.path.join(cluster_jpg_dir, jpg_name)
            shutil.copy(cat_jpg, cluster_jpg)

def get_args():
    parser = argparse.ArgumentParser(description='select pictures of clusters')
    parser.add_argument('-fi', dest='cluster_txt_dir', 
                        help='cluster txt dir', default=None, type=str)
    parser.add_argument('-in', dest='cat_jpg_dir', 
                        help='category jpg dir', default=None, type=str)
    parser.add_argument('-out', dest='cluster_jpg_root', 
                        help='output pictures for each cluster', default=None, type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args  

if __name__=='__main__':
    
    tic = time.clock()
    global cluster_jpg_root, cat_jpg_dir
    
    args = get_args()
    cluster_txt_dir = args.cluster_txt_dir
    cat_jpg_dir = args.cat_jpg_dir
    cluster_jpg_root = args.cluster_jpg_root

    cluster_txt_list = GetFileList(cluster_txt_dir, ['.rois.txt'])
    pool = Pool(int(cpu_count()*7/8))
    pool.map(get_pic_of_cluster, cluster_txt_list)

    toc = time.clock()
    print('running time:{:.3f}'.format(toc-tic))


