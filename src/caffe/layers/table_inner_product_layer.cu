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
void TableInnerProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
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
                  unsigned int index = (unsigned int)this->blobs_[bi]->gpu_data()[k];
                  tmp_blobs_[bi]->mutable_gpu_data()[k] = quant_table_.at(index);
              }
	  }
     tmp_blobs_updated_ = true;
    }

    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    const Dtype* weight = this->tmp_blobs_[0]->gpu_data();
    caffe_gpu_gemm<Dtype>(CblasNoTrans, transpose_ ? CblasNoTrans : CblasTrans,
          M_, N_, K_, (Dtype)1.,bottom_data, weight, (Dtype)0., top_data);
    if (bias_term_) {
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
          bias_multiplier_.gpu_data(),this->tmp_blobs_[1]->gpu_data(), (Dtype)1., top_data);
    }
}

template <typename Dtype>
void TableInnerProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
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
                    unsigned int index = (unsigned int)this->blobs_[bi]->gpu_data()[k];
		    tmp_blobs_[bi]->mutable_gpu_data()[k] = quant_table_.at(index);
	        }
	    }
            tmp_blobs_updated_ = true;
       }

       if (this->param_propagate_down_[0]) {
	    const Dtype* top_diff = top[0]->gpu_diff();
	    const Dtype* bottom_data = bottom[0]->gpu_data();

	    // Gradient with respect to weight
	    if (transpose_) {
		caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
			  K_, N_, M_,
			  (Dtype)1., bottom_data, top_diff,
			  (Dtype)1., this->tmp_blobs_[0]->mutable_gpu_diff());
	    } else {
		 caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
			  N_, K_, M_,
			  (Dtype)1., top_diff, bottom_data,
			  (Dtype)1., this->tmp_blobs_[0]->mutable_gpu_diff());
	   }
       }

      if (bias_term_ && this->param_propagate_down_[1]) {
         const Dtype* top_diff = top[0]->gpu_diff();
         // Gradient with respect to bias
         caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
            bias_multiplier_.gpu_data(), (Dtype)1.,
            this->tmp_blobs_[1]->mutable_gpu_diff());
      }

       //calculate gradients for cluster centers
      if (this->param_propagate_down_[0]) {
	 const Dtype* tmp_blobs_diff = this->tmp_blobs_[0]->gpu_diff();
         size_t weight_mul = 1;
	 for (size_t i = 0; i < this->blobs_[0]->shape().size(); i++) {
		 weight_mul *= this->blobs_[0]->shape()[i];
	 }
         for(int i=0; i < weight_mul; i++){
		int index = (int)this->blobs_[0]->gpu_data()[i];
		this->quant_table_.at(index)+=tmp_blobs_diff[i];
	 }	
         //memset(this->blobs_[0]->mutable_cpu_diff(), 0, this->blobs_[0]->count()*sizeof(Dtype));
	 //memset(this->int_blobs_[0]->mutable_cpu_diff(), 0, this->int_blobs_[0]->count()*sizeof(int));
		
       }

       if (bias_term_ && this->param_propagate_down_[1]) {
	  const Dtype* tmp_blobs_diff = this->tmp_blobs_[1]->gpu_diff();
          size_t weight_mul = 1;
	  for (size_t i = 0; i < this->blobs_[1]->shape().size(); i++) {
		 weight_mul *= this->blobs_[1]->shape()[i];
	  }
          for(int i=0; i < weight_mul; i++){
		int index = (int)this->blobs_[1]->gpu_data()[i];
		this->quant_table_.at(index)+=tmp_blobs_diff[i];
	  }
	//memset(this->blobs_[1]->mutable_cpu_diff(), 0, this->blobs_[1]->count()*sizeof(Dtype));
	//memset(this->int_blobs_[1]->mutable_cpu_diff(), 0, this->int_blobs_[1]->count()*sizeof(int));
       }


      if (propagate_down[0]) {
         const Dtype* top_diff = top[0]->gpu_diff();
        // Gradient with respect to bottom data
        if(transpose_){
           caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
              M_, K_, N_,
              (Dtype)1., top_diff, this->tmp_blobs_[0]->gpu_data(),
              (Dtype)0., bottom[0]->mutable_gpu_diff());
        } else {
          caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
              M_, K_, N_,
             (Dtype)1., top_diff, this->tmp_blobs_[0]->gpu_data(),
             (Dtype)0., bottom[0]->mutable_gpu_diff());
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

INSTANTIATE_LAYER_GPU_FUNCS(TableInnerProductLayer);

}  // namespace caffe
