#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void InnerProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_output = this->layer_param_.inner_product_param().num_output();
  bias_term_ = this->layer_param_.inner_product_param().bias_term();
  transpose_ = this->layer_param_.inner_product_param().transpose();
  N_ = num_output;
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize the weights
    vector<int> weight_shape(2);
    if (transpose_) {
      weight_shape[0] = K_;
      weight_shape[1] = N_;
    } else {
      weight_shape[0] = N_;
      weight_shape[1] = K_;
    }
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.inner_product_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      vector<int> bias_shape(1, N_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.inner_product_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  /************ For network pruning ***************/
  if(this->blobs_.size()==2 && (this->bias_term_)){
    this->masks_.resize(2);
    // Intialize and fill the weightmask & biasmask
    this->masks_[0].reset(new Blob<Dtype>(this->blobs_[0]->shape()));
    shared_ptr<Filler<Dtype> > weight_mask_filler(GetFiller<Dtype>(
        this->layer_param_.inner_product_param().weight_mask_filler()));
    weight_mask_filler->Fill(this->masks_[0].get());
    this->masks_[1].reset(new Blob<Dtype>(this->blobs_[1]->shape()));
    shared_ptr<Filler<Dtype> > bias_mask_filler(GetFiller<Dtype>(
        this->layer_param_.inner_product_param().bias_mask_filler()));
    bias_mask_filler->Fill(this->masks_[1].get());    
  }  
  else if(this->blobs_.size()==1 && (!this->bias_term_)){
    this->masks_.resize(1);	  
    // Intialize and fill the weightmask
    this->masks_[0].reset(new Blob<Dtype>(this->blobs_[0]->shape()));
    shared_ptr<Filler<Dtype> > bias_mask_filler(GetFiller<Dtype>(
        this->layer_param_.inner_product_param().bias_mask_filler()));
    bias_mask_filler->Fill(this->masks_[0].get());      
  }   
   
  // Intialize the tmp tensor 
  this->weight_tmp_.Reshape(this->blobs_[0]->shape());
  this->bias_tmp_.Reshape(this->blobs_[1]->shape());  
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  const int new_K = bottom[0]->count(axis);
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters.";
  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  M_ = bottom[0]->count(0, axis);
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);
  top_shape[axis] = N_;
  top[0]->Reshape(top_shape);
  // Set up the bias multiplier
  if (bias_term_) {
    vector<int> bias_shape(1, M_);
    bias_multiplier_.Reshape(bias_shape);
    caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  
  /************ For network pruning ***************/
  Dtype* weightMask = this->masks_[0]->mutable_cpu_data(); 
  Dtype* weightTmp = this->weight_tmp_.mutable_cpu_data();
  const Dtype* bias = NULL;
  Dtype* biasMask = NULL;  
  Dtype* biasTmp = NULL;

  Dtype Thres0,Thres1;
  if (this->bias_term_) {
    bias = this->blobs_[1]->mutable_cpu_data(); 
    biasMask = this->masks_[1]->mutable_cpu_data();
    biasTmp = this->bias_tmp_.mutable_cpu_data();
  }
   
  if (this->phase_ == TRAIN){	
		for (unsigned int k = 0;k < this->blobs_[0]->count(); ++k) {
			if (weightMask[k]==1 && fabs(weight[k])<=Thres0)
				weightMask[k] = 0;
			else if (weightMask[k]==0 && fabs(weight[k])>Thres1)
				weightMask[k] = 1;
		}	
		if (this->bias_term_) {       
			for (unsigned int k = 0;k < this->blobs_[1]->count(); ++k) {
				if (biasMask[k]==1 && fabs(bias[k])<=Thres0)
					biasMask[k] = 0;
				else if (biasMask[k]==0 && fabs(bias[k])>Thres1)
					biasMask[k] = 1;
			}    
		} 
	}    
	
  // Calculate the current (masked) weight and bias
	for (unsigned int k = 0;k < this->blobs_[0]->count(); ++k) {
		weightTmp[k] = weight[k]*weightMask[k];
	}
	if (this->bias_term_){
		for (unsigned int k = 0;k < this->blobs_[1]->count(); ++k) {
			biasTmp[k] = bias[k]*biasMask[k];
		}
	} 
	
	// Forward calculation with (masked) weight and bias 
  caffe_cpu_gemm<Dtype>(CblasNoTrans, transpose_ ? CblasNoTrans : CblasTrans,
      M_, N_, K_, (Dtype)1.,
      bottom_data, weightTmp, (Dtype)0., top_data);
  if (bias_term_) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        bias_multiplier_.cpu_data(),biasTmp, (Dtype)1., top_data);
  }
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype* weightMask = this->masks_[0]->cpu_data();
	Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
	// Gradient with respect to weight
	for (unsigned int k = 0;k < this->blobs_[0]->count(); ++k) {
		weight_diff[k] = weight_diff[k]*weightMask[k];
	}	
    if (transpose_) {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          K_, N_, M_,
          (Dtype)1., bottom_data, top_diff,
          (Dtype)1., weight_diff);
    } else {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          (Dtype)1., top_diff, bottom_data,
          (Dtype)1., weight_diff);
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->cpu_diff();
	const Dtype* biasMask = this->masks_[1]->cpu_data();
    Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
    // Gradient with respect to bias
    for (unsigned int k = 0;k < this->blobs_[1]->count(); ++k) {
		bias_diff[k] = bias_diff[k]*biasMask[k];
	}	
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.cpu_data(), (Dtype)1.,bias_diff);
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
	const Dtype* weightTmp = this->weight_tmp_.cpu_data();
    // Gradient with respect to bottom data
    if (transpose_) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, weightTmp,
          (Dtype)0., bottom[0]->mutable_cpu_diff());
    } else {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, weightTmp,
          (Dtype)0., bottom[0]->mutable_cpu_diff());
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(InnerProductLayer);
#endif

INSTANTIATE_CLASS(InnerProductLayer);
REGISTER_LAYER_CLASS(InnerProduct);

}  // namespace caffe
