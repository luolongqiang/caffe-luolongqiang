#**********************
#date:2016-08-22
#author:luolongqiang
#**********************

import os
from multiprocessing import Pool,cpu_count

def Experiment(k):
    print('crop pictures for cat'+str(k)+' ...')
    cat_txt_path = 'SampleRois/sampled_rois_by_cat/RoIsInCat'+str(k)+'.txt'
    input_jpg_root = '/export/sdtes/wangcm/intelli_fridge/fast-rcnn/data/fridge'                                                
    output_jpg_dir = 'SampleRois/sampled_rois_by_cat_pic/cat'+str(k)                                        
    if not os.path.exists(output_jpg_dir):                                                                                                                   
       os.mkdir(output_jpg_dir)                                                                                                                             
    os.system('python crop_pic_of_category.py -fi '+ cat_txt_path + ' -in ' + input_jpg_root + ' -out ' + output_jpg_dir)    

if __name__=='__main__':
 
    pool = Pool(int(cpu_count()*7/8))
    pool.map(Experiment, range(21))
    
   
