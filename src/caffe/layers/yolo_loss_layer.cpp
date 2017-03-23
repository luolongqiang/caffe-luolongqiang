#include <vector>
#include <math.h>

#include "caffe/layers/yolo_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void YoloLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
    const vector<Blob<Dtype>*>& top) {
    LossLayer<Dtype>::LayerSetUp(bottom, top);

    YoloLossParameter yolo_param = this->layer_param_.yolo_loss_param();
    grid_side_ = yolo_param.grid_side();
    class_count_ = yolo_param.classes();
    bbox_count_ = yolo_param.bbox_count();
    output_num_ = grid_side_ * grid_side_ * (bbox_count_*5+class_count_);
    int truth_num_ = grid_side_ * grid_side_ * (5+class_count_); 
    noobject_scale_ = yolo_param.noobject_scale();
    coord_scale_ = yolo_param.coord_scale();
    class_scale_ = yolo_param.class_scale();
    object_scale_ = yolo_param.object_scale();
    sqrt_ = yolo_param.sqrt();
    rescore_ = yolo_param.rescore();

    CHECK_EQ(bottom[0]->shape(1), output_num_) 
        << "The prediction and output should have the same number.";
    CHECK_EQ(bottom[1]->shape(1), truth_num_)
        << "The truth and output should have the same number.";
}

template <typename Dtype>
void YoloLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    LossLayer<Dtype>::Reshape(bottom, top);
    diff_.ReshapeLike(*bottom[0]);
    caffe_set(diff_.count(), static_cast<Dtype>(0), diff_.mutable_cpu_data());
}

template <typename Dtype>
void YoloLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    //compute loss
    Dtype loss = 0;
    float avg_obj = 0.0;
    float avg_cat = 0.0;
    float avg_iou = 0.0;
    float avg_anyobj = 0.0;
    int count = 0;
    const Dtype* yolo_output = bottom[0]->cpu_data();
    const Dtype* yolo_truth = bottom[1]->cpu_data();
    Dtype* delta = diff_.mutable_cpu_data();
    const int batch_size = bottom[0]->num();
    const int grid_num = grid_side_ * grid_side_;
    const int grid_pred = (class_count_ + 5 * bbox_count_) * grid_num;
    for (int b = 0; b < batch_size; b++) {
        DLOG(INFO) << "Batch: " << b;
        for (int i = 0; i < grid_num; i++) {
            //no object loss
            int truth_offset = (b*grid_num+i) * (5+class_count_);
            int is_obj = yolo_truth[truth_offset];
            for (int j = 0; j < bbox_count_; j++) {
                int obj_offset = b*grid_pred + class_count_*grid_num + i*bbox_count_ + j;     //is object
                delta[obj_offset] = noobject_scale_ * (0 - yolo_output[obj_offset]);
                loss += noobject_scale_ * pow(yolo_output[obj_offset], 2);
                avg_anyobj += yolo_output[obj_offset];
            }

            if (!is_obj) {
                continue;
            }

            //class loss
            int class_index = b*grid_pred + i*class_count_;
            for (int j = 0; j < class_count_; j++) {
                delta[class_index+j] = class_scale_ * (yolo_truth[truth_offset+1+j] - yolo_output[class_index+j]);
                loss += class_scale_ * pow(yolo_truth[truth_offset+1+j] - yolo_output[class_index+j], 2);
                if (yolo_truth[truth_offset+1+j]) {
                    avg_cat += yolo_output[class_index+j];
                }
            }

            //find best bbox
            Dtype truth_box[4];
            memcpy(truth_box, yolo_truth+truth_offset+1+class_count_, 4*sizeof(Dtype));
            truth_box[0] /= grid_side_;
            truth_box[1] /= grid_side_;

            int best_index = -1;
            float best_iou = 0;
            float best_rmse = 20;
            for (int j = 0; j < bbox_count_; j++) {
                int bbox_index = b*grid_pred + (class_count_+bbox_count_)*grid_num + (i*bbox_count_+j)*4;
                Dtype out_box[4];
                memcpy(out_box, yolo_output+bbox_index, 4*sizeof(Dtype));
                out_box[0] /= grid_side_;
                out_box[1] /= grid_side_;
                if (sqrt_) {
                    out_box[2] = out_box[2]*out_box[2];
                    out_box[3] = out_box[3]*out_box[3];
                }
                float iou = BoxIou(truth_box, out_box);
                float rmse = BoxRmse(truth_box, out_box);
                if (best_iou > 0 || iou > 0) {
                    if (iou > best_iou) {
                        best_iou = iou;
                        best_index = j;
                    }
                }else {
                    if (rmse < best_rmse) {
                        best_rmse = rmse;
                        best_index = j;
                    }
                }
            }
            int bbox_index = b*grid_pred + (class_count_+bbox_count_)*grid_num + (i*bbox_count_+best_index)*4;
            int truth_box_offset = truth_offset + 1 + class_count_;

            Dtype out_box[4];
            memcpy(out_box, yolo_output+bbox_index, 4*sizeof(Dtype));
            DLOG(INFO) << "Size of Dtype: " << sizeof(Dtype);
            DLOG(INFO) << "Size of float: " << sizeof(float);
            out_box[0] /= grid_side_;
            out_box[1] /= grid_side_;
            if (sqrt_) {
                out_box[2] = out_box[2]*out_box[2];
                out_box[3] = out_box[3]*out_box[3];
            }
            float iou = BoxIou(truth_box, out_box);

            //object loss
            int obj_offset = b*grid_pred + class_count_*grid_num + i*bbox_count_ + best_index;
            loss -= noobject_scale_ * pow(yolo_output[obj_offset], 2);
            loss += object_scale_ * pow(1-yolo_output[obj_offset], 2);
            avg_obj += yolo_output[obj_offset];
            delta[obj_offset] = object_scale_ * (1-yolo_output[obj_offset]);
            
            if (rescore_) {
                delta[obj_offset] = object_scale_ * (iou-yolo_output[obj_offset]);
            }

            //bbox loss
            for (int j = 0; j < 4; j++) {
                delta[bbox_index+j] = coord_scale_*(yolo_truth[truth_box_offset+j] - yolo_output[bbox_index+j]);
            }
            if (sqrt_) {
                delta[bbox_index+2] = coord_scale_*(sqrt(yolo_truth[truth_box_offset+2]) - yolo_output[bbox_index+2]);
                delta[bbox_index+3] = coord_scale_*(sqrt(yolo_truth[truth_box_offset+3]) - yolo_output[bbox_index+3]);
            }

            loss += pow(1-iou, 2);
            avg_iou += iou;
            ++count;
            DLOG(INFO) << "Truth box x: " << truth_box[0];
            DLOG(INFO) << "Truth box y: " << truth_box[1];
            DLOG(INFO) << "Truth box w: " << truth_box[2];
            DLOG(INFO) << "Truth box h: " << truth_box[3];
            DLOG(INFO) << "Out box x: " << out_box[0];
            DLOG(INFO) << "Out box y: " << out_box[1];
            DLOG(INFO) << "Out box w: " << out_box[2];
            DLOG(INFO) << "Out box h: " << out_box[3];
            DLOG(INFO) << "IoU: " << iou;
            DLOG(INFO) << "Avg_iou: " << avg_iou;
            DLOG(INFO) << "Count: " << count;
        }
    }
    for (int i = 0; i < bottom[0]->count(); i++) {
        DLOG(INFO) << "yolo output: " << i << " " << yolo_output[i];
    }
    for (int i = 0; i < bottom[1]->count(); i++) {
        DLOG(INFO) << "yolo truth:" << i << " " << yolo_truth[i];
    }
    for (int i = 0; i < bottom[0]->count(); i++) {
        DLOG(INFO) << "diff: " << i << " " << diff_.cpu_data()[i];
    }
    //Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
    top[0]->mutable_cpu_data()[0] = avg_iou/count;
    //LOG(INFO) << "Avg IoU: " << avg_iou/count;
}

template <typename Dtype>
void YoloLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    caffe_cpu_axpby(
        bottom[0]->count(),
        Dtype(-1),
        diff_.cpu_data(),
        Dtype(0),
        bottom[0]->mutable_cpu_diff());

    for (int i = 0; i < bottom[0]->count(); i++) {
        DLOG(INFO) << "bottom 0 diff: " << i << " " << bottom[0]->mutable_cpu_diff()[i];
    }
}

template <typename Dtype>
Dtype YoloLossLayer<Dtype>::BoxIou(Dtype* box_a, Dtype* box_b){
    Dtype w = Overlap(box_a[0], box_a[2], box_b[0], box_b[2]);
    Dtype h = Overlap(box_a[1], box_a[3], box_b[1], box_b[3]);
    if (w < 0 || h < 0) {
        return Dtype(0);
    }
    Dtype intersection = w*h;
    Dtype box_union = box_a[2]*box_a[3] + box_b[2]*box_b[3] - intersection;
    return Dtype(intersection/box_union);
}

template <typename Dtype>
Dtype YoloLossLayer<Dtype>::BoxRmse(Dtype* box_a, Dtype* box_b){
    Dtype sum = 0;
    for (int i = 0; i < 4; i++) {
        sum += pow(box_a[i]-box_b[i], 2);
    }
    return Dtype(sqrt(sum));
}

template <typename Dtype>
Dtype YoloLossLayer<Dtype>::Overlap(Dtype x1, Dtype w1, Dtype x2, Dtype w2){
    Dtype l1 = x1 - w1/2;
    Dtype l2 = x2 - w2/2;
    Dtype left = l1 > l2 ? l1 : l2;
    Dtype r1 = x1 + w1/2;
    Dtype r2 = x2 + w2/2;
    Dtype right = r1 < r2 ? r1 : r2;
    return Dtype(right - left);
}

INSTANTIATE_CLASS(YoloLossLayer);
REGISTER_LAYER_CLASS(YoloLoss);

}