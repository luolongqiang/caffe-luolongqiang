# -*- coding:UTF-8 -*-
import os, sys, time

input_root = "data/shoes/362-test/chirdren"
for input_dir in os.listdir(input_root):
    if not input_dir.endswith('-test'):
        print '-------------', input_dir, '-------------'
        time.sleep(5)
        os.system("python python/shoes_demo/shoes_demo.py \
                   -mc models/shoes_demo/classify_deploy.prototxt \
                   -wc models/shoes_demo/shoes_classify.caffemodel \
                   -md models/shoes_demo/detection_deploy.prototxt \
                   -wd models/shoes_demo/shoes_detection.caffemodel" + \
                   " -i " + input_root + "/" + input_dir + \
                   " -o " + input_root + "/" + input_dir + "-test")