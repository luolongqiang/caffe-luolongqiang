#ifdef USE_CUDNN
#include <vector>
#include <math.h>

#include "caffe/layers/cudnn_conv_layer.hpp"

namespace caffe {

__global__ void sync_conv_groups() { }

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();

    // Forward through cuDNN in parallel over groups.
    for (int g = 0; g < this->group_; g++) {
      // Filters.
      CUDNN_CHECK(cudnnConvolutionForward(handle_[g],
            cudnn::dataType<Dtype>::one,
            bottom_descs_[i], bottom_data + bottom_offset_ * g,
            filter_desc_, weight + this->weight_offset_ * g,
            conv_descs_[i],
            fwd_algo_[i], workspace[g], workspace_fwd_sizes_[i],
            cudnn::dataType<Dtype>::zero,
            top_descs_[i], top_data + top_offset_ * g));
	    /* added start */
		Dtype top_shift = 1.0/pow(2, global_top_scale); //W*FM
		Dtype bias_shift = 1.0/pow(2, global_bias_scale); //add bias
		
		static const void *tmp_top_use = (const void*)(&top_shift);		
		static const void *tmp_bias_use = (const void*)(&bias_shift);		
		/* added end */	
		
      // Bias.
      if (this->bias_term_) {
        const Dtype* bias_data = this->blobs_[1]->gpu_data();
       CUDNN_CHECK(cudnnAddTensor(handle_[g],
             tmp_bias_use,
             bias_desc_, bias_data + bias_offset_ * g,
             tmp_top_use,
             top_descs_[i], top_data + top_offset_ * g));
      }
    }

    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
    sync_conv_groups<<<1, 1>>>();
  }
}
