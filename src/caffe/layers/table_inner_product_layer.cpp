/*
All modification made by Intel Corporation: © 2016 Intel Corporation

All contributions by the University of California:
Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
All rights reserved.

All other contributions:
Copyright (c) 2014, 2015, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/table_inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void TableInnerProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
     tmp_blobs_updated_ = false;
     table_size_ = this->layer_param_.table_inner_product_param().table_size();
     const int num_output = this->layer_param_.table_inner_product_param().num_output();
     bias_term_ = this->layer_param_.table_inner_product_param().bias_term();
     transpose_ = this->layer_param_.table_inner_product_param().transpose();
     N_ = num_output;
     const int axis = bottom[0]->CanonicalAxisIndex(
          this->layer_param_.table_inner_product_param().axis());
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
          this->tmp_blobs_.resize(2);
        } else {
          this->blobs_.resize(1); 
          this->tmp_blobs_.resize(1);
        }
        //this->quant_table_.resize(1);
        // Initialize the weights
        weight_shape = vector<int>(2);
   
        if (transpose_) {
          weight_shape[0] = K_;
          weight_shape[1] = N_;
        } else {
          weight_shape[0] = N_;
          weight_shape[1] = K_;
        }

        //this->tmp_blobs_[0].reset(new Blob<Dtype>(weight_shape));
        this->blobs_[0].reset(new Blob<Dtype>(weight_shape));

        //shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        //   this->layer_param_.table_inner_product_param().weight_filler()));
        //weight_filler->Fill(this->quant_blobs_[0].get());

        // Resetting the codebook table
        //vector<int> table_shape(1, table_size_);
        //this->quant_table_.reset(new Blob<Dtype>(table_shape));	
        this->quant_table_.resize(table_size_);
        // If necessary, intiialize and fill the bias term
        if (bias_term_) {
          bias_shape = vector<int>(1, N_);
          //this->tmp_blobs_[1].reset(new Blob<Dtype>(bias_shape));
          // Resetting the int bias blob
          this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
        }
    }  // parameter initialization
    this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void TableInnerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    // Figure out the dimensions
    const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.table_inner_product_param().axis());
 
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
void TableInnerProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    if (!tmp_blobs_updated_) {
          CHECK_EQ(table_size_, this->blobs_[0]->get_quant_table_size());
          for(int i=0; i<table_size_;i++)
             quant_table_.at(i)=this->blobs_[0]->quant_table_data(i);
          
          tmp_blobs_.resize(this->blobs_.size());
          for (size_t bi = 0; bi < this->blobs_.size(); bi++) {
              tmp_blobs_[bi].reset(new Blob<Dtype>(this->blobs_[bi]->shape()));
	      size_t weight_mul = 1;
	      for (size_t i = 0; i < this->blobs_[bi]->shape().size(); i++) {
		  weight_mul *= this->blobs_[bi]->shape()[i];
	      }
              for (size_t k = 0; k < weight_mul; k++) {
	          // Putting the value into the table
                  unsigned int index = (unsigned int)this->blobs_[bi]->cpu_data()[k];
                  tmp_blobs_[bi]->mutable_cpu_data()[k] = quant_table_.at(index);
              }
	  }
     tmp_blobs_updated_ = true;
    }

    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    const Dtype* weight = this->tmp_blobs_[0]->cpu_data();
    caffe_cpu_gemm<Dtype>(CblasNoTrans, transpose_ ? CblasNoTrans : CblasTrans,
          M_, N_, K_, (Dtype)1.,bottom_data, weight, (Dtype)0., top_data);
    if (bias_term_) {
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
          bias_multiplier_.cpu_data(),this->tmp_blobs_[1]->cpu_data(), (Dtype)1., top_data);
    }
}

template <typename Dtype>
void TableInnerProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

      if (!tmp_blobs_updated_) {
            CHECK_EQ(table_size_, this->blobs_[0]->get_quant_table_size());
            for(int i=0; i<table_size_;i++)
               quant_table_.at(i)=this->blobs_[0]->quant_table_data(i);

	    tmp_blobs_.resize(this->blobs_.size());
            for (size_t bi = 0; bi < this->blobs_.size(); bi++) {
		tmp_blobs_[bi].reset(new Blob<Dtype>(this->blobs_[bi]->shape()));
                
                size_t weight_mul = 1;
	        for (size_t i = 0; i < this->blobs_[bi]->shape().size(); i++) {
		  weight_mul *= this->blobs_[bi]->shape()[i];
	        }
	        for (size_t k = 0; k < weight_mul; k++) {
		    // Putting the value into the table
                    unsigned int index = (unsigned int)this->blobs_[bi]->cpu_data()[k];
		    tmp_blobs_[bi]->mutable_cpu_data()[k] = quant_table_.at(index);
	        }
	    }
            tmp_blobs_updated_ = true;
       }

       if (this->param_propagate_down_[0]) {
	    const Dtype* top_diff = top[0]->cpu_diff();
	    const Dtype* bottom_data = bottom[0]->cpu_data();

	    // Gradient with respect to weight
	    if (transpose_) {
		caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
			  K_, N_, M_,
			  (Dtype)1., bottom_data, top_diff,
			  (Dtype)1., this->tmp_blobs_[0]->mutable_cpu_diff());
	    } else {
		 caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
			  N_, K_, M_,
			  (Dtype)1., top_diff, bottom_data,
			  (Dtype)1., this->tmp_blobs_[0]->mutable_cpu_diff());
	   }
       }

      if (bias_term_ && this->param_propagate_down_[1]) {
         const Dtype* top_diff = top[0]->cpu_diff();
         // Gradient with respect to bias
         caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
            bias_multiplier_.cpu_data(), (Dtype)1.,
            this->tmp_blobs_[1]->mutable_cpu_diff());
      }

      //calculate gradients for cluster centers
      if (this->param_propagate_down_[0]) {
	 const Dtype* tmp_blobs_diff = this->tmp_blobs_[0]->cpu_diff();
         size_t weight_mul = 1;
	 for (size_t i = 0; i < this->blobs_[0]->shape().size(); i++) {
		 weight_mul *= this->blobs_[0]->shape()[i];
	 }
         for(int i=0; i < weight_mul; i++){
		int index = (int)this->blobs_[0]->cpu_data()[i];
		this->quant_table_.at(index)+=tmp_blobs_diff[i];
	 }	
         //memset(this->blobs_[0]->mutable_cpu_diff(), 0, this->blobs_[0]->count()*sizeof(Dtype));
	 //memset(this->int_blobs_[0]->mutable_cpu_diff(), 0, this->int_blobs_[0]->count()*sizeof(int));
		
       }

       if (bias_term_ && this->param_propagate_down_[1]) {
	  const Dtype* tmp_blobs_diff = this->tmp_blobs_[1]->cpu_diff();
          size_t weight_mul = 1;
	  for (size_t i = 0; i < this->blobs_[1]->shape().size(); i++) {
		 weight_mul *= this->blobs_[1]->shape()[i];
	  }
          for(int i=0; i < weight_mul; i++){
		int index = (int)this->blobs_[1]->cpu_data()[i];
		this->quant_table_.at(index)+=tmp_blobs_diff[i];
	  }
	//memset(this->blobs_[1]->mutable_cpu_diff(), 0, this->blobs_[1]->count()*sizeof(Dtype));
	//memset(this->int_blobs_[1]->mutable_cpu_diff(), 0, this->int_blobs_[1]->count()*sizeof(int));
       }

      if (propagate_down[0]) {
         const Dtype* top_diff = top[0]->cpu_diff();
        // Gradient with respect to bottom data
        if(transpose_){
           caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
              M_, K_, N_,
              (Dtype)1., top_diff, this->tmp_blobs_[0]->cpu_data(),
              (Dtype)0., bottom[0]->mutable_cpu_diff());
        } else {
          caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
              M_, K_, N_,
             (Dtype)1., top_diff, this->tmp_blobs_[0]->cpu_data(),
             (Dtype)0., bottom[0]->mutable_cpu_diff());
        }

       //memset(this->tmp_blobs_[0]->mutable_cpu_diff(), 0, this->tmp_blobs_[0]->count()*sizeof(Dtype));
       //memset(this->tmp_blobs_[1]->mutable_cpu_diff(), 0, this->tmp_blobs_[1]->count()*sizeof(Dtype));
     }
     for(int i=0; i<table_size_;i++)
     {
          this->blobs_[0]->set_quant_table_data(i,quant_table_.at(i));
          this->blobs_[1]->set_quant_table_data(i,quant_table_.at(i));
     }
}

#ifdef CPU_ONLY
STUB_GPU(TableInnerProductLayer);
#endif

INSTANTIATE_CLASS(TableInnerProductLayer);

REGISTER_LAYER_CLASS(TableInnerProduct);

}  // namespace caffe
