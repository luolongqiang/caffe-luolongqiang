import os, sys, time
import cv2
import argparse
import numpy as np
from numpy import array
from pandas import DataFrame
from multiprocessing import Pool, cpu_count
import caffe
caffe.set_mode_cpu()

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

def get_real_labels(real_txt, w_img, h_img):
    labels = {'body':[0,0,0,0],'face':[0,0,0,0]}
    name_dict = {0:'body',1:'face'}
    with open(real_txt, 'r') as fi:
        for line in fi:
            obj = map(float, line.strip().split())
            labels[name_dict[obj[0]]] = [obj[1]*w_img, obj[2]*h_img, obj[3]*w_img, obj[4]*h_img]
    return labels

def get_pred_labels(pred_txt):
    labels = {'body':[0,0,0,0,0],'face':[0,0,0,0,0]}
    with open(pred_txt, 'r') as fi:
        for line in fi:
            obj = line.strip().split()
            cls = obj[0]
            prob = float(obj[5])
            if labels[cls][-1] < prob:
                labels[cls] = [float(obj[1]), float(obj[2]), float(obj[3]), float(obj[4]), prob]
    return labels

def show_results(img, pred_labels, w_img, h_img, output_img_file):
    img_cp = img.copy()
    img_cp *= 255
    imshow = True
    for ele in pred_labels.keys():
        x = int(pred_labels[ele][0])
        y = int(pred_labels[ele][1])
        w = int(pred_labels[ele][2]/2)
        h = int(pred_labels[ele][3]/2)
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
            cv2.putText(img_cp, ele + ' : %.2f' % pred_labels[ele][-1], (xmin + 5, ymin - 7),
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

def make_evaluation(arguments):
    input_img_file, real_txt, pred_txt, output_img_file = arguments[0], arguments[1], arguments[2], arguments[3]
    img = caffe.io.load_image(input_img_file)  # load the image using caffe io
    #img = cv2.imread(input_img_file)
    w_img, h_img = img.shape[1], img.shape[0]
    real_labels = get_real_labels(real_txt, w_img, h_img)
    pred_labels = get_pred_labels(pred_txt)
    img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    show_results(img_cv, pred_labels, w_img, h_img, output_img_file)
   
    img_name = input_img_file[input_img_file.rfind('/')+1:]
    print '****** ' + img_name + ' ******'
    results = [] 
    for ele in ['body', 'face']:
        iou, left_error, top_error, right_error, bottom_error, ave_error, prob = compared_two_boxes(real_labels[ele], pred_labels[ele])
        print  'for '+ ele + ', results:'
        print 'IoU:{:.3f}, left error:{:.3f}, top error:{:.3f}, right error:{:.3f}, bottom error:{:.3f}, average error:{:.3f}, prob:{:.3f}'\
              .format(iou, left_error, top_error, right_error, bottom_error, ave_error, prob)
        results.append([img_name, iou, left_error, top_error, right_error, bottom_error, ave_error, prob])
    return results

def output_to_csv(results, output_eval_dir, label):
    df = DataFrame({'image':results[:,0],\
                    'IoU':results[:,1],\
                    'left error':results[:,2],\
                    'top error':results[:,3],\
                    'right error':results[:,4],\
                    'bottom error':results[:,5],\
                    'average error':results[:,6],\
                    'probability':results[:,7]})    
    df = df[['image', 'IoU', 'left error', 'top error', 'right error', 'bottom error', 'average error', 'probability']]
    df.to_csv(output_eval_dir+'/'+label+'_eval_results.csv', index = False)

def get_args():
    parser = argparse.ArgumentParser(description='output the results of prediction') 
    parser.add_argument('-t', dest='test_txt',
        help='test.txt includes pathes of tesing images', default=None, type=str)
    parser.add_argument('-p', dest='pred',
        help='input directory includes .txt files of predicted labels', default=None, type=str)
    parser.add_argument('-o', dest='out_img',
        help='output directory path for images with bboxes', default=None, type=str)
    parser.add_argument('-e', dest='out_eval',
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
    
    test_txt = args.test_txt
    pred_txt_dir = args.pred
    output_img_dir = args.out_img
    output_eval_dir = args.out_eval
    cpu_num = args.cpu_num

    if not os.path.exists(output_img_dir):
       os.makedirs(output_img_dir)
    if not os.path.exists(output_eval_dir):
       os.makedirs(output_eval_dir)
    input_img_list, real_txt_list, pred_txt_list, output_img_list = \
       get_file_list(test_txt, pred_txt_dir, output_img_dir)

    tic = time.clock() 
    print "the number of samples for evaluation:"+str(len(input_img_list))
    pool = Pool(cpu_num)
    all_results = pool.map(make_evaluation, zip(input_img_list, real_txt_list, pred_txt_list, output_img_list))
    all_results = array(all_results)
    body_results = all_results[:,0]
    face_results = all_results[:,1]
    output_to_csv(body_results, output_eval_dir, 'body')
    output_to_csv(face_results, output_eval_dir, 'face')
    toc = time.clock()
    print 'running time:{:.3f} seconds'.format(toc - tic)
