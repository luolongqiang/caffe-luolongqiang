#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/yolo_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
class YoloLossLayerTest : public CPUDeviceTest<Dtype> {
    
protected:
    YoloLossLayerTest()
        : blob_bottom_data_(new Blob<Dtype>()),
          blob_bottom_label_(new Blob<Dtype>()),
          blob_top_(new Blob<Dtype>()) {
        
        vector<int> shape(2);
        shape[0] = 1;
        shape[1] = 588;
        blob_bottom_data_->Reshape(shape);
        blob_bottom_label_->Reshape(shape);
        
        blob_bottom_vec_.push_back(blob_bottom_data_);
        blob_bottom_vec_.push_back(blob_bottom_label_);
        blob_top_vec_.push_back(blob_top_);
    }

    virtual void FillBottoms() {
        FillerParameter filler_param;
        GaussianFiller<Dtype> filler(filler_param);
        filler.Fill(this->blob_bottom_data_);
        filler.Fill(this->blob_bottom_label_);
    }

    virtual ~YoloLossLayerTest() {
        delete blob_bottom_data_;
        delete blob_bottom_label_;
        delete blob_top_;
    }

    Blob<Dtype>* const blob_bottom_data_;
    Blob<Dtype>* const blob_bottom_label_;
    Blob<Dtype>* const blob_top_;
    vector<Blob<Dtype>*> blob_bottom_vec_;
    vector<Blob<Dtype>*> blob_top_vec_;

};

TYPED_TEST_CASE(YoloLossLayerTest, TestDtypes);

TYPED_TEST(YoloLossLayerTest, TestSetup) {
    LayerParameter layer_param;
    YoloLossParameter* yolo_loss_param = layer_param.mutable_yolo_loss_param();
    yolo_loss_param->set_grid_side(7);
    yolo_loss_param->set_classes(2);
    yolo_loss_param->set_bbox_count(2);
    YoloLossLayer<TypeParam> layer(layer_param);

    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
}

TYPED_TEST(YoloLossLayerTest, TestForward) {
    LayerParameter layer_param;
    YoloLossParameter* yolo_loss_param = layer_param.mutable_yolo_loss_param();
    yolo_loss_param->set_grid_side(7);
    yolo_loss_param->set_classes(2);
    yolo_loss_param->set_bbox_count(2);
    YoloLossLayer<TypeParam> layer(layer_param);

    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
}

}