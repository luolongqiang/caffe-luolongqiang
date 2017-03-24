********************************************************* 基于yolo的目标检测 ***********************************
1. 分割训练集和测试集: train.txt, test.txt 
其中train.txt中每行存放的是图片路径, labels/目录下存放的是.txt文件， 每个.txt文件内容存放的是一张图片的label:(class, x, y, w, h)
2. 训练： 
(1). 在文件models/yolo/solver.prototxt中设置net文件路径，以及各个参数
(2). 在net文件：models/yolo/yolo_small_train_val.prototxt中修改数据路径以及output_num等
(3). python python/yolo/train.py -s models/yolo/solver.prototxt -m models/yolo/yolo_ft_out/body_yolo27_132w.caffemodel -i 1 2>&1 | tee models/yolo/yolo_ft_out/out.log
3. 测试
方式一：python python/yolo/yolo_test.py -m models/yolo/yolo.prototxt -w models/yolo/yolo_ft_out/_iter_132w.caffemodel -t data/body_det/test.txt -o models/yolo/results/132w_pred_labels -c 16 -p 0.4
方式二：python python/yolo/yolo_test.py -m models/yolo/yolo.prototxt -w models/yolo/yolo_ft_out/_iter_132w.caffemodel -i data/body_det/JPEGImages -o models/yolo/results/132w_pred_labels -c 16 -p 0.4
4. 评估
python python/yolo/yolo_eval.py -t data/body_det/test.txt -p models/yolo/results/132w_pred_labels -o models/yolo/results/132w_bbox_imgs -e models/yolo/results/132w_evals -c 16
5. 分析
python python/yolo/output_error_cases.py -c models/yolo/results/132w_evals/body_eval_results.csv -i models/yolo/results/132w_bbox_imgs -o models/yolo/results/132w_evals/error_cases_iou_0.5 -t 0.5
****************************************************************************************************************


******************************************************** product: 多任务学习 ************************************
0. 转换格式：
python python/product/json_to_txt.py -j data/product/product_list.json -t data/product/product_list.txt
1. 统计分析：
python python/product/count_label.py -t data/product/product_list.txt -c data/product/category_count.csv -k category
2. 分割数据集：train, val, test
python python/product/get_partition_consider_all.py -t data/product/product_list.txt -p data/product
3. 训练：
(1). 生成log文件：
python python/product/train.py -s models/multi_task_vgg16_bn/solver.prototxt -m models/multi_task_vgg16_bn/vgg16_bn_ft_out/_iter_20w.caffemodel -i 1 2>&1 | tee models/multi_task_vgg16_bn/vgg16_bn_ft_out/out.log
(2). 重启训练：
./build/tools/caffe train --solver=models/multi_task_vgg16_bn/solver.prototxt --snapshot=models/multi_task_vgg16_bn/vgg16_bn_ft_out/_iter_20w.solverstate 2>&1 | tee models/multi_task_vgg16_bn/vgg16_bn_ft_out/out.log
(3). 画出loss vs iters曲线: 
scp luolongqiang@IP:服务器  本地
./tools/extra/plot_training_log.py.example 0 models/multi_task_vgg16_bn/vgg16_bn_ft_out/test_accuracy.png models/multi_task_vgg16_bn/vgg16_bn_ft_out/out.log
4. 测试
python python/product/test.py -m models/multi_task_vgg16_bn/deploy_larger.prototxt -w models/multi_task_vgg16_bn/vgg16_bn_ft_out/_iter_20w.caffemodel -i data/product/multi_task_test_label.txt -o models/multi_task_vgg16_bn/results/20w.txt -c 16
5. 评估
python python/product/evaluation.py -r data/product/multi_task_test_label.txt -p models/multi_task_vgg16_bn/results/8w.txt -isout 1
***************************************************************************************************************


********************************************************* deepFashion: 多标签预测 ******************************
1. 分割数据集：train, val, test
python python/deepFashion/get_style_multilabel.py -t data/deepFashion/Anno/list_attr_cloth.txt -m data/deepFashion/Anno/list_attr_img.txt -p data/deepFashion/Eval/list_eval_partition.txt -o data/deepFashion
2. 统计分析：
python python/deepFashion/get_style_informations.py -i data/deepFashion/style_train_multilabel.txt -s data/deepFashion/style_name_list.txt -o data/deepFashion/style_informations.csv -c data/deepFashion/count_style_results.csv
3. 训练：
(1). 生成log文件：
python python/deepFashion/train.py -s models/style_vgg19_bn_bn/solver.prototxt -m models/style_vgg19_bn_bn/vgg19_bn_cvgj_iter_320000.caffemodel -i 1 2>&1 | tee models/style_vgg19_bn_bn/vgg19_bn_bn_ft_out/out.log
(2). 重启训练：
./build/tools/caffe train --solver=models/style_vgg19_bn/solver.prototxt --snapshot=models/style_vgg19_bn/vgg19_bn_ft_out/_iter_20000.solverstate 2>&1 | tee models/style_vgg19_bn/vgg19_bn_ft_out/out.log
(3). 画出loss vs iters曲线: 
scp luolongqiang@IP:服务器  本地
./tools/extra/plot_training_log.py.example 0 models/style_vgg19_bn//vgg19_bn_ft_out/test_accuracy.png models/style_vgg19_bn/vgg19_bn_ft_out/out.log
4. 测试
python python/deepFashion/test_multilabel.py -m models/style_vgg19_bn/deploy.prototxt -w models/style_vgg19_bn/crop_vgg19_bn_ft_out/_iter_20000.caffemodel -i data/deepFashion/style_test_multilabel.txt -o models/style_vgg19_bn/results/2w.txt  -c 16 -t 0.0 -k 5 -mode test
5. 评估
python python/deepFashion/evaluation_multilabel.py -r data/deepFashion/style_test_multilabel.txt -p models/style_vgg19_bn/results/2w.txt -l models/style_vgg19_bn/results/2w.csv  -c 16 -t 0.0 -k 20
***************************************************************************************************************


********************************************************* product: shoes分类 **********************************
2. 分割数据集：train, val, test
python python/shoes/get_shoes_information.py -i data/shoes/img -o data/shoes/shoes_count.csv -p data/shoes
3. 训练：
(1). 生成log文件：
python python/shoes/train.py -s models/shoes_vgg19_bn/solver.prototxt -m models/shoes_vgg19_bn/vgg19_bn_ft_out/_iter_10000.caffemodel -i 1 2>&1 | tee models/shoes_vgg19_bn/vgg19_bn_ft_out/out.log
(2). 重启训练：
./build/tools/caffe train --solver=models/shoes_vgg19_bn/solver.prototxt --snapshot=models/shoes_vgg19_bn/vgg19_bn_ft_out/_iter_10000.solverstate 2>&1 | tee models/shoes_vgg19_bn/vgg19_bn_ft_out/out.log
(3). 画出loss vs iters曲线: 
scp luolongqiang@IP:服务器  本地
./tools/extra/plot_training_log.py.example 0 models/shoes_vgg19_bn/vgg19_bn_ft_out/test_loss.png models/shoes_vgg19_bn/vgg19_bn_ft_out/out.log
4. 测试
python python/shoes/test.py -m models/shoes_vgg19_bn/deploy.prototxt -w models/shoes_vgg19_bn/vgg19_bn_ft_out/_iter_10000.caffemodel -i data/shoes/shoes_val_test_label.txt -o models/shoes_vgg19_bn/results/shoes_1w.txt -s models/shoes_vgg19_bn/results/shoes_1w.csv -c 16 -n 10
python python/shoes/test.py -m models/shoes_vgg19_bn/deploy.prototxt -w models/shoes_vgg19_bn/vgg19_bn_ft_out/_iter_10000.caffemodel -i data/shoes/shoes_val_test_label.txt -c 16 -n 10
5. 评估
python python/product/evaluation.py -r data/product/shoes_test_label.txt -p models/shoes_vgg19_bn/results/1w.txt -l models/shoes_vgg19_bn/results/1w.csv  -c 16 -n 10
*****************************************************************************************************************
caffemodels: https://pan.baidu.com/s/1eSGFFlK
Netscope: http://ethereon.github.io/netscope/#/editor
