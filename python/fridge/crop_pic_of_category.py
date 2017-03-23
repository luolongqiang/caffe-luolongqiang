#**********************
#date:2016-08-10
#author:luolongqiang
#**********************

import sys,os,argparse,time,cv2

def crop_pic(input_txt_path, input_jpg_root, output_jpg_dir):
    fi = open(input_txt_path)
    while True:
        s = fi.readline()
        if not s:
            break
        line = s.strip().split(' ')
        temp = line[0].split('/')
        jpg_path = os.path.join(input_jpg_root,temp[0],'jpg',temp[1])
        x, y, z, w = float(line[3]), float(line[4]), float(line[5]), float(line[6])
        x, y, z, w = int(x), int(y), int(z), int(w)
        im = cv2.imread(jpg_path)
        crop_im = im[y:w, x:z]
        output_jpg_path =os.path.join(output_jpg_dir,temp[0]+'-'+temp[1].split('.')[0]+'-'+line[1]+'.jpg') 
        cv2.imwrite(output_jpg_path, crop_im) 
    
def GetArgs():
    parser = argparse.ArgumentParser(description='crop pictures of category')
    
    parser.add_argument('-fi', dest='category_txt', 
                        help='category txt file', default=None, type=str)
    parser.add_argument('-in', dest='input_jpg_root', 
                        help='input picture root', default=None, type=str)
    parser.add_argument('-out', dest='category_jpg_dir', 
                        help='output picture of category', default=None, type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args    
    
if __name__=='__main__':
    
    tic = time.clock()
    
    args = GetArgs()
    input_txt_path = args.category_txt
    input_jpg_root = args.input_jpg_root
    output_jpg_dir = args.category_jpg_dir
    crop_pic(input_txt_path, input_jpg_root, output_jpg_dir)
    
    toc = time.clock()
    print('running time:{:.3f} seconds'.format(toc-tic))
 

