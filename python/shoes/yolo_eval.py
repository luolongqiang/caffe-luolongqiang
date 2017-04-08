import numpy as np
from numpy import array
from pandas import DataFrame
from multiprocessing import Pool, cpu_count
import os, sys, time, argparse
import caffe, cv2
caffe.set_mode_cpu()

# python python/shoes/yolo_eval.py -t data/shoes/test.txt -p models/shoes/results/pred_labels_1w -o models/shoes/results/bbox_imgs_1w -e models/shoes/results/eval_1w.csv -c 16

def get_file_list(test_txt, pred_txt_dir, output_img_dir):
    input_img_list = []
    real_txt_list = []
    pred_txt_list = []
    output_img_list = []
    with open(test_txt, 'r') as fi:
         for line in fi:
             line = line.strip() 
             tmp = line[0:line.rfind('/')]
             input_dir = tmp[0:tmp.rfind('/')]
             img_name = line[line.rfind('/')+1:].split('.')[0]
             input_img_list.append(line)
             real_txt_list.append(input_dir+'/labels/'+img_name+'.txt')
             pred_txt_list.append(pred_txt_dir+'/'+img_name+'.txt')
             output_img_list.append(output_img_dir+'/'+img_name+'.jpg')
    return input_img_list, real_txt_list, pred_txt_list, output_img_list

def make_evaluation(arguments):
    input_img_file, real_txt, pred_txt, output_img_file = arguments
    img = caffe.io.load_image(input_img_file)  # load the image using caffe io
    w_img, h_img = img.shape[1], img.shape[0]
    real_labels = get_real_labels(real_txt, w_img, h_img)
    pred_labels = get_pred_labels(pred_txt)
    img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    show_results(img_cv, pred_labels, w_img, h_img, output_img_file)
    
    img_name = input_img_file[input_img_file.rfind('/')+1:]
    iou, left_error, top_error, right_error, bottom_error, ave_error, prob = \
        compared_two_boxes(real_labels, pred_labels)
    print 'IoU:{:.3f}, average error:{:.3f}, prob:{:.3f}'.format(iou, ave_error, prob)

    return [img_name, iou, left_error, top_error, right_error, bottom_error, ave_error, prob]

def get_real_labels(real_txt, w_img, h_img):
    fi = open(real_txt, 'r')
    line = fi.read()
    obj = map(float, line.strip().split())
    labels = [obj[1]*w_img, obj[2]*h_img, obj[3]*w_img, obj[4]*h_img]
    return labels
    
def get_pred_labels(pred_txt):
    labels = [0, 0, 0, 0, 0]
    fi = open(pred_txt, 'r')
    line = fi.read()
    if line:
        labels = map(float, line.strip().split()[1:])
    return labels

def show_results(img, pred_labels, w_img, h_img, output_img_file):
    img_cp = img.copy()
    img_cp *= 255
    imshow = True
    x = int(pred_labels[0])
    y = int(pred_labels[1])
    w = int(pred_labels[2]/2)
    h = int(pred_labels[3]/2)
    xmin = x - w
    xmax = x + w
    ymin = y - h
    ymax = y + h
    if xmin < 0:
        xmin = 0
    if ymin < 0:
        ymin = 0
    if xmax > w_img:
        xmax = w_img
    if ymax > h_img:
        ymax = h_img
    if imshow:
        cv2.rectangle(img_cp, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.rectangle(img_cp, (xmin, ymin - 20), (xmax, ymin), (125, 125, 125), -1)
        cv2.putText(img_cp, str(pred_labels[-1])[:4], (xmin + 5, ymin - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.imwrite(output_img_file, img_cp)

def compared_two_boxes(box1, box2):
    lt_xmin = min(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
    lt_ymin = min(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
    lt_xmax = max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
    lt_ymax = max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
    rb_xmin = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2])
    rb_ymin = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3])
    rb_xmax = max(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2])
    rb_ymax = max(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3])  
    w_inter = rb_xmin - lt_xmax
    h_inter = rb_ymin - lt_ymax
    w_union = rb_xmax - lt_xmin
    h_union = rb_ymax - lt_ymin
    left_dist = lt_xmax - lt_xmin
    top_dist = lt_ymax - lt_ymin
    right_dist = rb_xmax - rb_xmin
    bottom_dist = rb_ymax - rb_ymin
    if w_inter <= 0 or  h_inter<= 0:
        inter_area = 0
    else:
        inter_area = w_inter * h_inter
    union_area = box1[2] * box1[3] + box2[2] * box2[3] - inter_area
    if union_area == 0:
        iou = 0
    else:
        iou = inter_area / union_area
    if w_union == 0:
        left_error = 0
        right_error = 0
    else:
        left_error =  left_dist / w_union
        right_error = right_dist / w_union
    if h_union == 0:
        top_error = 0
        bottom_error = 0
    else:
        top_error = top_dist / h_union
        bottom_error = bottom_dist / h_union        
    ave_error = (left_error + top_error + right_error + bottom_error) / 4

    prob = box2[4]
    if(sum(array(box1))==0 and sum(array(box2)[0:4])==0):
        iou = 1
        prob = 1
    return iou, left_error, top_error, right_error, bottom_error, ave_error, prob

def output_to_csv(results, output_csv):
    df = DataFrame({'image':results[:,0],\
                    'IoU':results[:,1],\
                    'left error':results[:,2],\
                    'top error':results[:,3],\
                    'right error':results[:,4],\
                    'bottom error':results[:,5],\
                    'average error':results[:,6],\
                    'probability':results[:,7]})    
    df = df[['image', 'IoU', 'left error', 'top error', 'right error', 
         'bottom error', 'average error', 'probability']]
    df.to_csv(output_csv, index = False)

def get_args():
    parser = argparse.ArgumentParser(description='output the results of prediction') 
    parser.add_argument('-t', dest='test_txt',
        help='test.txt includes pathes of tesing images', default=None, type=str)
    parser.add_argument('-p', dest='pred',
        help='input directory includes .txt files of predicted labels', default=None, type=str)
    parser.add_argument('-o', dest='out_img',
        help='output directory path for images with bboxes', default=None, type=str)
    parser.add_argument('-e', dest='out_csv',
        help='output results of evaluation', default=None, type=str)
    parser.add_argument('-c', dest='cpu_num',
        help='used cpu numbers for multiprocessing', default=6, type=int)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = get_args()
    
    test_txt        = args.test_txt
    pred_txt_dir    = args.pred
    output_img_dir  = args.out_img
    output_csv      = args.out_csv
    cpu_num         = args.cpu_num

    tic = time.time()

    if not os.path.exists(output_img_dir):
       os.makedirs(output_img_dir)

    input_img_list, real_txt_list, pred_txt_list, output_img_list = \
       get_file_list(test_txt, pred_txt_dir, output_img_dir)
     
    print "the number of samples for evaluation:"+str(len(input_img_list))

    pool = Pool(cpu_num)
    results = pool.map(make_evaluation, \
       zip(input_img_list, real_txt_list, pred_txt_list, output_img_list))
    '''
    results = []
    for arguments in zip(input_img_list, real_txt_list, pred_txt_list, output_img_list):
        results.append(make_evaluation(arguments))
    '''

    results = array(results)
    if output_csv: 
        output_to_csv(results, output_csv)

    ave_IoU   = np.mean(map(float, results[:,1]))
    ave_error = np.mean(map(float, results[:,6]))
    ave_prob  = np.mean(map(float, results[:,7]))
    print "*******************************************************"
    print "average IoU:{:.3f}, average error:{:.3f}, average prob:{:.3f}".format(ave_IoU, ave_error, ave_prob)
    print "*******************************************************"

    toc = time.time()
    print 'running time:{:.3f} seconds'.format(toc - tic)
