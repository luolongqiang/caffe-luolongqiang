#ifndef CAFFE_YOLO_LOSS_LAYER_HPP_
#define CAFFE_YOLO_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class YoloLossLayer : public LossLayer<Dtype> {

public:
    explicit YoloLossLayer(const LayerParameter& param)
        : LossLayer<Dtype>(param), diff_() {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "YoloLoss"; }

protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);

    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

    Dtype BoxIou(Dtype* box_a, Dtype* box_b);

    Dtype BoxRmse(Dtype* box_a, Dtype* box_b);

    Dtype Overlap(Dtype x1, Dtype w1, Dtype x2, Dtype w2);
    
    int grid_side_;
    int class_count_;
    int bbox_count_;
    int output_num_;
    float noobject_scale_;
    float class_scale_;
    float object_scale_;
    float coord_scale_;
    bool sqrt_;
    bool rescore_;

    Blob<Dtype> diff_;
};

}

#endif 