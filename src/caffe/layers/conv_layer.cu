#include <vector>

#include "caffe/layers/conv_layer.hpp"

namespace caffe {
template <typename Dtype>
__global__ void CCMaskCalc(const int n, const Dtype* wb,
    Dtype* mask, Dtype thres0, Dtype thres1) {
  CUDA_KERNEL_LOOP(index, n) {
    if (mask[index]==1 && fabs(wb[index])<=thres0) 
      mask[index] = 0;
    else if (mask[index]==0 && fabs(wb[index])>thres1)
      mask[index] = 1;
  }
}

template <typename Dtype>
__global__ void CCMaskApply(const int n, const Dtype* wb,
    const Dtype* mask, Dtype* wb_t) {
  CUDA_KERNEL_LOOP(index, n) {
    wb_t[index] = wb[index] * mask[index];    
  }
}
template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  const Dtype* bias = NULL;
  Dtype* biasMask = NULL;
  Dtype* biasTmp = NULL; 
  Dtype* weightMask=NULL;
  Dtype* weightTmp=NULL;
  if (this->bias_term_)   
    bias = this->blobs_[1]->mutable_gpu_data(); 
  if(this->pruning_)
  { 
  Dtype thres0=this->layer_param_.pruning_thres();
  Dtype thres1=thres0;
  weightMask = this->masks_[0]->mutable_gpu_data();
  weightTmp = this->weight_tmp_.mutable_gpu_data();
  if (this->bias_term_) {  
    biasMask = this->masks_[1]->mutable_gpu_data();
    biasTmp = this->bias_tmp_.mutable_gpu_data();
  }
  //if (this->phase_ == TRAIN){
	// Calculate the weight mask and bias mask with probability 
      CCMaskCalc<Dtype><<<CAFFE_GET_BLOCKS(this->blobs_[0]->count()),
        CAFFE_CUDA_NUM_THREADS>>>( this->blobs_[0]->count(), weight, 
        weightMask, thres0, thres1);
      CUDA_POST_KERNEL_CHECK;    
      if (this->bias_term_) {   
        CCMaskCalc<Dtype><<<CAFFE_GET_BLOCKS(this->blobs_[1]->count()),
          CAFFE_CUDA_NUM_THREADS>>>( this->blobs_[1]->count(), bias, 
          biasMask, thres0, thres1);
        CUDA_POST_KERNEL_CHECK; 
      }    
  //}   

  // Calculate the current (masked) weight and bias
  CCMaskApply<Dtype><<<CAFFE_GET_BLOCKS(this->blobs_[0]->count()),
    CAFFE_CUDA_NUM_THREADS>>>( this->blobs_[0]->count(), weight, weightMask, weightTmp);
  CUDA_POST_KERNEL_CHECK;
  if (this->bias_term_) {  
    CCMaskApply<Dtype><<<CAFFE_GET_BLOCKS(this->blobs_[1]->count()),
      CAFFE_CUDA_NUM_THREADS>>>( this->blobs_[1]->count(), bias, biasMask, biasTmp);
    CUDA_POST_KERNEL_CHECK;  
  }
  }
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    for (int n = 0; n < this->num_; ++n) {
      if(this->pruning_)
      {
        this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weightTmp,
          top_data + n * this->top_dim_);
      }
      else
      {
        this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      }  
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();
        if(this->pruning_)
        this->forward_gpu_bias(top_data + n * this->top_dim_, biasTmp);
        else
        this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  const Dtype* weightTmp = NULL;  	
  const Dtype* weightMask = NULL;
  if(this->pruning_)
  {
    weightTmp = this->weight_tmp_.gpu_data();  	
    weightMask = this->masks_[0]->gpu_data();
  }
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      if(this->pruning_){
      const Dtype* biasMask = this->masks_[1]->gpu_data();
      CCMaskApply<Dtype><<<CAFFE_GET_BLOCKS(this->masks_[1]->count()),
        CAFFE_CUDA_NUM_THREADS>>>( this->masks_[1]->count(), bias_diff, biasMask, bias_diff);
      CUDA_POST_KERNEL_CHECK;
      }
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      if(this->pruning_){
      CCMaskApply<Dtype><<<CAFFE_GET_BLOCKS(this->masks_[0]->count()),
        CAFFE_CUDA_NUM_THREADS>>>( this->masks_[0]->count(), weight_diff, weightMask, weight_diff);
      CUDA_POST_KERNEL_CHECK; 
      }
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          if(this->pruning_){
            this->backward_gpu_gemm(top_diff + n * this->top_dim_, weightTmp,
              bottom_diff + n * this->bottom_dim_);
          }
          else
          {
            this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_); 
          }
        }
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionLayer);

}  // namespace caffe
