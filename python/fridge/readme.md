1. Code: GetOptimalThreshold.py
Cmd line: (1). python GetOptimalThreshold.py -cfg cfg_GetOptimalThreshold.cfg 
          (2). python GetOptimalThreshold.py LABEL_TXT_ROOT PREDICTED_TXT_ROOT 
               Example: python  GetOptimalThreshold.py  ../fast-rcnn/data/fridge/test_0614  ../fast-rcnn/output/default/20cls_test_0614/vgg16_fast_rcnn_iter_60000/intermediate
Output to current dir: OptimalThreshold.csv

2. Code: FilterWithThreshold.py
Cmd line: (1). python FilterWithThreshold.py -cfg cfg_FilterWithThreshold.cfg
          (2). python FilterWithThreshold.py  INPUT_TXT_ROOT  OUTPUT_TXT_ROOT  OptimalThreshold.csv
               Example: python FilterWithThreshold.py  ../fast-rcnn/output/default/20cls_test_0614/vgg16_fast_rcnn_iter_60000/intermediate   ../fast-rcnn/output/default/20cls_test_0614/vgg16_fast_rcnn_iter_60000/final_data_20cls  OptimalThreshold.csv

3. Code: EvalFinal.py
Cmd line: (1). python EvalFinal.py -cfg cfg_EvalFinal.cfg
          (2). python EvalFinal.py LABEL_TXT_ROOT PREDICTED_TXT_ROOT [-isbox 1]
               Example: python EvalFinal.py ../fast-rcnn/data/fridge/test_0614  ../fast-rcnn/output/default/20cls_test_0614/vgg16_fast_rcnn_iter_60000/final_data_20cls -isbox 1
Output to current dir: ClassInfsOfEvaluation.csv && ResultsOfEvaluation.csv

4. Code: Visualize.py
Cmd line: (1). python Visualize.py -cfg cfg_Visualize.cfg
          (2). python Visualize.py real  JPG&TXT-ROOT  PREDICTED_TXT_ROOT  OUTPUT_MERGED_JPG_ROOT
               Example: python Visualize.py  ../fast-rcnn/data/fridge/test_0614  ../fast-rcnn/output/default/20cls_test_0614/vgg16_fast_rcnn_iter_60000/final_data_20cls ../fast-rcnn/output/default/20cls_test_0614/vgg16_fast_rcnn_iter_60000/merged_jpg_20cls

5. Code: EvalRpn.py
Cmd line: (1). python EvalRpn.py -cfg cfg_EvalRpn.cfg [-t 0.5]
          (2). python EvalRpn.py REAL_LABELS_ROOT PREDICTED_LABELS_ROOT [-t 0.5]
          Example: python EvalRpn.py /export/sdtes/luolongqiang/intelli_fridge/fast-rcnn/data/fridge/test_0614 /export/sdtes/zhouming/rpn/VGG16_0.7_0.0/test_0614 [-t 0.5]

6. Code: crop_pic_for_error_cases.py
Cmd line: python  crop_pic_for_error_cases.py  -intxt  ERROR_CASE_TXT_PATH  -injpg  PREDICTED_JPG_PATH  –out  OUTPUT_DIRECTORY  [-mode crop (or draw)]  
Example: python  crop_pic_for_error_cases.py  -intxt  eval/error_cases.txt  -injpg  eval/jpg  -out  eval/error_cases_0826  [-mode crop( or draw)]

7. Code: get_optimal_wh_ratio.py
Cmd line: python get_optimal_wh_ratio.py -in REAL_LABELS_ROOT -out DIR_OF_RESULTS
Example: python get_optimal_wh_ratio.py -in /export/sdtes/luolongqiang/intelli_fridge/fast-rcnn/data/fridge/train -out results

8. Code: transform_labels.py
Cmd line: python transform_labels.py -mode TO_HSX (or TO_ORIGIN) -in INPUT_TXT_ROOT -out OUTPUT_TXT_ROOT
Mode1 Example: python transform_labels.py -mode to_hsx -in /export/sdtes/luolongqiang/intelli_fridge/fast-rcnn/data/fridge/train -out /export/sdtes/luolongqiang/intelli_fridge/fast-rcnn/data/fridge/train26 [-t 0.7]
Mode2 Example: python transform_labels.py -mode to_origin -in /export/sdtes/luolongqiang/intelli_fridge/fast-rcnn/output/default/26cls_test_0614/vgg16_fast_rcnn_iter_60000/intermediate -out /export/sdtes/luolongqiang/intelli_fridge/fast-rcnn/output/default/26cls_test_0614/vgg16_fast_rcnn_iter_60000/intermediate20

9. Code: crop_pic_of_category.py
Cmd line: python crop_pic_of_category.py -fi CATEGORY_TXT_DIR -in INPUT_JPG_ROOT -out CATEGORY_JPG_DIR
Example: python crop_pic_of_category.py -fi /export/sdtes/zouyu/intelli_fridge/fast-rcnn/output/rois_feats_tools/SampleRois/sampled_rois_by_cat/RoIsInCat1.txt -in /export/sdtes/wangcm/intelli_fridge/fast-rcnn/data/fridge -out /export/sdtes/zouyu/intelli_fridge/fast-rcnn/output/rois_feats_tools/SampleRois/sampled_rois_by_cat_pic/cat1

同时运行21个category：python script_crop_pic_of_category.py

10. Code: select_pic_of_clusters.py
Cmd line: python select_pic_of_clusters.py -fi CLUSTER_TXT_DIR -in CATEGORY_JPG_DIR -out CLUSTER_JPG_ROOT
Example: python select_pic_of_clusters.py -fi /export/sdtes/zouyu/intelli_fridge/fast-rcnn/output/rois_feats_tools/SampleRois/cat19 -in /export/sdtes/zouyu/intelli_fridge/fast-rcnn/output/rois_feats_tools/SampleRois/sampled_rois_by_cat_pic/cat19 -out /export/sdtes/zouyu/intelli_fridge/fast-rcnn/output/rois_feats_tools/SampleRois/cat19_cluster0to63_pic

11. Code: eval_topk_labels.py
Cmd line: (1). python eval_topk_labels.py -cfg cfg_eval_topk_labels.py
          (2). python eval_topk_labels.py LABEL_TXT_ROOT PREDICTED_TXT_ROOT [-topk 2]
          Example: python eval_topk_labels.py /export/sdtes/luolongqiang/intelli_fridge/fast-rcnn/data/fridge/test_0614 /root/wangcm/intelli_fridge/py-faster-rcnn/output/faster_rcnn_alt_opt/20cls_test_0614.cfg/vgg16_jd_fridge_fast_rcnn_only_stage_1_faster_rcnn_alt_opt_jd_fridge_0.5_rpn_nms.yml_iter_60000/intermediate_softmax -topk 2 

12. Code: get_box_range_of_test.py
Cmd line: python get_box_range_of_test.py -in REAL_LABELS_ROOT -out OUTPUT_ROOT [-isDraw 1]
Example: python get_box_range_of_test.py -in /export/sdtes/luolongqiang/intelli_fridge/fast-rcnn/data/fridge/test_0614 -out /export/sdtes/luolongqiang/intelli_fridge/eval/box_range [-isDraw 1]

13. Code: filter_with_box_range.py
Cmd line: python filter_with_box_range.py -in INPUT_PREDICTED_LABELS_ROOT -out OUTPUT_PREDICRED_LABELS_ROOT -csv BOX_RANGE.csv [-IoU 0.5]
Example: python filter_with_box_range.py -in /export/sdtes/luolongqiang/intelli_fridge/fast-rcnn/output/default/20cls_test_0614/vgg16_fast_rcnn_iter_60000/final_data_20cls -out /export/sdtes/luolongqiang/intelli_fridge/fast-rcnn/output/default/20cls_test_0614/vgg16_fast_rcnn_iter_60000/final_data_20cls_filtered_with_box_range -csv /export/sdtes/luolongqiang/intelli_fridge/eval/box_range/20160614_2_13283/box_range.csv -IoU 0.5

14. Code: DrawImage.py
Cmd line: python DrawImage.py -cfg cfg_DrawImage.cfg

