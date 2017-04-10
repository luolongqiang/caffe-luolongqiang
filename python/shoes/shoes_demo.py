# -*- coding:UTF-8 -*-
import numpy as np
from multiprocessing import Pool
import os, sys, time, argparse, shutil
import caffe, cv2
caffe.set_mode_cpu()

# python python/shoes_demo/shoes_demo.py -mc models/shoes_demo/classify_deploy.prototxt -wc models/shoes_demo/shoes_classify.caffemodel -md models/shoes_demo/detection_deploy.prototxt -wd models/shoes_demo/shoes_detection.caffemodel -i data/shoes/362-test/JPEGImages1 -o data/shoes/362-test/JPEGImages1-test
# python python/shoes_demo/shoes_demo.py -mc models/shoes_demo/classify_deploy.prototxt -wc models/shoes_demo/shoes_classify.caffemodel -md models/shoes_demo/detection_deploy.prototxt -wd models/shoes_demo/shoes_detection.caffemodel -i data/shoes/362-test/chirdren/ECZ7113013W -o data/shoes/362-test/chirdren/ECZ7113013W-test

def shoes_demo(cls_model, cls_weight, det_model, det_weight, input_dir, output_root):
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    
    cls_net = caffe.Net(cls_model, cls_weight, caffe.TEST)
    det_net = caffe.Net(det_model, det_weight, caffe.TEST)

    for img_file in os.listdir(input_dir):
        if img_file.endswith('.jpg'):
            print '*****************************************************************'
            print img_file
            input_img_file = os.path.join(input_dir, img_file)
            img = caffe.io.load_image(input_img_file) # RGB and /255
            p_label = make_classify(cls_net, img)
            output_dir = os.path.join(output_root, str(p_label))
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            output_img_file = os.path.join(output_dir, img_file)
            if p_label in range(1, 11):
                yolo_box, prob = make_detection(det_net, img)
                img = cv2.imread(input_img_file) # BGR
                canny_box = get_bbox_by_mask(img)
                output_final_box(canny_box, yolo_box, img, p_label, prob, output_img_file)
            else:
                shutil.copy(input_img_file, output_img_file)
                print '(p_label, top, right, bottom, left):({}, {}, {}, {}, {})'.\
                   format(p_label, None, None, None, None)
    #end

def make_classify(net, img):
    width, height = net.blobs['data'].data.shape[2:]
    inputs = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_raw_scale('data', 256.0) 
    transformer.set_channel_swap('data', (2,1,0))
    out = net.forward(data = np.asarray([transformer.preprocess('data', inputs)]))
    probs = out['prob'][0]
    p_label = np.argmax(probs)+1   
    return p_label

def make_detection(net, img):
    inputs = img.copy()
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    out = net.forward_all(data = np.asarray([transformer.preprocess('data', inputs)]))
    top, right, bottom, left, prob = \
        interpret_output(out['result'][0], img.shape[1], img.shape[0])
    yolo_box = [top, right, bottom, left]
    return yolo_box, prob

def get_bbox_by_mask(img):
    # resize img to fit_img
    raw_w, raw_h = img.shape[1], img.shape[0]
    size = 300
    if raw_w >= raw_h:
        fit_w = int(float(size)*raw_w/raw_h)   
        fit_h = size
    else:
        fit_h = int(float(size)*raw_h/raw_w) 
        fit_w = size  
    fit_img = cv2.resize(img, (fit_w, fit_h), interpolation = cv2.INTER_LINEAR)

    # get fit_bbox
    top, right, bottom, left = get_bbox(fit_img, fit_w, fit_h)
    
    # resize fit_bbox to bbox
    top = int(float(top)*raw_h/fit_h)
    right = int(float(right)*raw_w/fit_w)
    bottom = int(float(bottom)*raw_h/fit_h)
    left = int(float(left)*raw_w/fit_w)

    canny_box = [top, right, bottom, left]

    return canny_box
    
def get_bbox(fit_img, fit_w, fit_h):
    # canny for bbox
    grayed = cv2.cvtColor(fit_img, cv2.COLOR_BGR2GRAY)
    grayed = cv2.GaussianBlur(grayed, (5, 5), 0)
    pix = np.mean([grayed[0, 0], grayed[0, fit_w/2], grayed[0, fit_w-1], \
        grayed[fit_h/2, 0], grayed[fit_h/2, fit_w-1], \
        grayed[fit_h-1, 0], grayed[fit_h-1, fit_w/2], grayed[fit_h-1, fit_w-1]])
    pix = 255 - int(pix)
    # 这个阈值告诉算法“什么程度的边界才算边缘”，阈值越大表示标准越严厉，提取到的边缘越少
    canny_img = cv2.Canny(grayed, pix + 5, pix + 35) 
    if np.max(canny_img) == 0:
        top    = 0
        right  = fit_w - 1
        bottom = fit_h - 1
        left   = 0
    else:
        linepix = np.where(canny_img == 255)
        top    = min(linepix[0])
        right  = max(linepix[1])
        bottom = max(linepix[0])
        left   = min(linepix[1])

    # grabcut for mask
    mask = np.zeros((fit_h, fit_w), np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (left, top, right - left, bottom - top) 
    cv2.grabCut(fit_img, mask, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==0)|(mask==2), 0, 1).astype('uint8')
    
    # get binary img with mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    th3 = cv2.dilate(canny_img, kernel)
    bin_img = cv2.bitwise_and(th3, th3, mask = mask2)
    
    # findContours for final bbox
    contours, dummy = cv2.findContours(bin_img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) 
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    if len(contours) > 0:
        box0 = cv2.boundingRect(contours[0])
        left = box0[0]
        top = box0[1]
        right = box0[0] + box0[2]
        bottom = box0[1] + box0[3]
        if len(contours) > 1:
            box1 = cv2.boundingRect(contours[1])
            if (box1[2]*box1[3]*1.0)/(box0[2]*box0[3]) >= 0.4:
                left = min(box0[0], box1[0])
                top = min(box0[1], box1[1])
                right = max(box0[0]+box0[2], box1[0]+box1[2])
                bottom = max(box0[1]+box0[3], box1[1]+box1[3])

    return top, right, bottom, left

def output_final_box(box1, box2, img, p_label, prob, output_img_file):
    # get iou value of two boxes (top, right, bottom, left)
    tb = min(box1[2], box2[2]) - min(box1[0], box2[0])
    lr = min(box1[1], box2[1]) - max(box1[3], box2[3])
    if tb < 0 or lr < 0:
        intersection = 0
    else:
        intersection = tb * lr
    area1 = (box1[1] - box1[3]) * (box1[2] - box1[0])
    area2 = (box2[1] - box2[3]) * (box2[2] - box2[0])
    union = (area1 + area2 - intersection) 
    if union == 0:
       iou_value = 0
    else:
       iou_value = intersection*1.0 / union

    # determine final box
    if iou_value >= 0.85:
        top, right, bottom, left = tuple(box1)
        fg = 'canny'
    else:
        top, right, bottom, left = tuple(box2)
        fg = str(prob)[:4]
    
    # draw box and output result
    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.rectangle(img, (left, top - 20), (right, top), (125, 125, 125), -1)
    cv2.putText(img, fg, (left+5, top-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    cv2.imwrite(output_img_file, img)

    print '(p_label, top, right, bottom, left):({}, {}, {}, {}, {})'.\
        format(p_label, top, right, bottom, left)

def interpret_output(output, w_img, h_img):
    num_box = 2
    num_class = 1
    grid_size = 7
    iou_threshold = 0.5
    index1 = grid_size**2*num_class
    index2 = index1+grid_size**2*num_box
    probs = np.zeros((grid_size, grid_size, num_box, num_class))
    class_probs = np.reshape(output[0:index1], (grid_size, grid_size, num_class))
    scales = np.reshape(output[index1:index2], (grid_size, grid_size, num_box))
    boxes = np.reshape(output[index2:], (grid_size, grid_size, num_box, 4))
    offset = np.reshape(np.array([np.arange(grid_size)]*(grid_size*num_box)), \
       (num_box, grid_size, grid_size))
    offset = np.transpose(offset, (1, 2, 0))

    # ------- get bbox -----------
    boxes[:, :, :, 0] += offset
    boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
    boxes[:, :, :, 0:2] = boxes[:, :, :, 0:2] / (grid_size*1.0)
    boxes[:, :, :, 2] = np.multiply(boxes[:, :, :, 2], boxes[:, :, :, 2])
    boxes[:, :, :, 3] = np.multiply(boxes[:, :, :, 3], boxes[:, :, :, 3])

    boxes[:, :, :, 0] *= w_img
    boxes[:, :, :, 1] *= h_img
    boxes[:, :, :, 2] *= w_img
    boxes[:, :, :, 3] *= h_img

    # ------ get confidence ------
    for i in range(num_box):
        for j in range(num_class):
            probs[:, :, i, j] = np.multiply(class_probs[:, :, j], scales[:, :, i])

    # --- confidence threshold ---
    filter_mat_probs = np.array(probs >= np.max(probs), dtype='bool')
    filter_mat_boxes = np.nonzero(filter_mat_probs)
    boxes_filtered = boxes[filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]
    probs_filtered = probs[filter_mat_probs]

    argsort = np.array(np.argsort(probs_filtered))[::-1]
    boxes_filtered = boxes_filtered[argsort][0]
    prob = probs_filtered[argsort][0]
    
    # get top, right, bottom, left
    x, y, w, h = tuple(boxes_filtered)
    left   = int(x - w/2)
    top    = int(y - h/2)
    right  = int(x + w/2)
    bottom = int(y + h/2)
    if left < 0:
        left = 0
    if top < 0:
        top = 0
    if right > w_img:
        right = w_img
    if bottom > h_img:
        bottom = h_img

    return top, right, bottom, left, prob

def get_args():
    parser = argparse.ArgumentParser(description='shoes classification and detection')
    parser.add_argument('-mc', dest='c_model', 
        help='classification network with format .prototxt', default=None, type=str)
    parser.add_argument('-wc', dest='c_weight',
        help='classification weights with format .caffemodel', default=None, type=str)
    parser.add_argument('-md', dest='d_model', 
        help='detection network with format .prototxt', default=None, type=str)
    parser.add_argument('-wd', dest='d_weight',
        help='detection weights with format .caffemodel', default=None, type=str)
    parser.add_argument('-i', dest='input_dir',
        help='input directory of images', default=None, type=str)
    parser.add_argument('-o', dest='output_root', 
        help='output root of images', default=None, type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args    

if __name__ == '__main__':

    args = get_args()

    cls_model   = args.c_model
    cls_weight  = args.c_weight
    det_model   = args.d_model
    det_weight  = args.d_weight
    input_dir   = args.input_dir
    output_root = args.output_root

    tic = time.time()
    shoes_demo(cls_model, cls_weight, det_model, det_weight, input_dir, output_root)
    toc = time.time()

    print('running time:{:.3f} seconds'.format(toc - tic))
    
