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

#include <cstring>
#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>

#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"
#ifdef USE_OPENCV
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif 

using std::ofstream;

using namespace caffe;  // NOLINT(build/namespaces)
using namespace std;
#ifdef USE_OPENCV

int main(int argc, char** argv) {
     FLAGS_alsologtostderr = 1;  // Print output to stderr (while still logging)
     ::google::InitGoogleLogging(argv[0]);
     if (argc != 7) {
        LOG(ERROR) << "Usage: "
        << "convert_inner_product_to_table <prototxt_in> <caffemodel_in> <prototxt_out> <caffemodel_out> <init_flags PP/RANDOM> <cluster_count>";
     return 1;
     }

     NetParameter caffemodel_net_param;
     NetParameter prototxt_net_param;
     string prototxt_in_filename(argv[1]);
     string caffemodel_in_filename(argv[2]);
     string prototxt_out_filename(argv[3]);
     string caffemodel_out_filename(argv[4]);
     string initType(argv[5]);
     int cluster_count = atoi(argv[6]);
     
     int flag;
     if(initType=="PP")
        flag=cv::KMEANS_PP_CENTERS;
     else
        flag=cv::KMEANS_RANDOM_CENTERS;
    
     ReadNetParamsFromTextFileOrDie(prototxt_in_filename, &prototxt_net_param);
     ReadNetParamsFromBinaryFileOrDie(caffemodel_in_filename, &caffemodel_net_param);
  
     //NetParameter targetNetParam;
     for (int i = 0; i < caffemodel_net_param.layer_size(); i++) {
	      LayerParameter* caffemodel_layerParam = caffemodel_net_param.mutable_layer(i);
              string caffemodel_layer_name = caffemodel_layerParam->name();
              LayerParameter* prototxt_layerParam = NULL;
		  
	      for (int j = 0; j < prototxt_net_param.layer_size(); j++) {
	         LayerParameter* other_prototxt_layer_param = prototxt_net_param.mutable_layer(j);
	         string prototxt_layer_name = other_prototxt_layer_param->name();

	         if (caffemodel_layer_name == prototxt_layer_name) {
		       prototxt_layerParam = other_prototxt_layer_param;
		       break;
	         }
	      }

	      if (prototxt_layerParam == NULL) {
	        LOG(WARNING) << "Layer not found: "<< caffemodel_layer_name;
	        return 2;
	      }

	      LOG(INFO) << caffemodel_layer_name;
	      if (caffemodel_layerParam->type() == "InnerProduct") {
                   const InnerProductParameter& innerProductParam = caffemodel_layerParam->inner_product_param();
                   LOG(INFO) << "Found InnerProduct layer. Converting to TableInnerProduct";

	           TableInnerProductParameter* caffemodel_tableInnerProductParam = caffemodel_layerParam->mutable_table_inner_product_param();
	           TableInnerProductParameter* prototxt_tableInnerProductParam = prototxt_layerParam->mutable_table_inner_product_param();
	           caffemodel_layerParam->set_type("TableInnerProduct");
	           prototxt_layerParam->set_type("TableInnerProduct");
                   caffemodel_tableInnerProductParam->set_num_output(innerProductParam.num_output());
	           prototxt_tableInnerProductParam->set_num_output(innerProductParam.num_output());
            
	           vector<float> weights;
                   for (size_t bi = 0; bi < caffemodel_layerParam->blobs_size(); bi++) {
			  size_t weight_mul = 1;
		          for (size_t i = 0; i < caffemodel_layerParam->blobs(bi).shape().dim_size(); i++) {
			            weight_mul *= caffemodel_layerParam->blobs(bi).shape().dim(i);
		          }
                       for (size_t k = 0; k < weight_mul; k++) {
                       weights.push_back(caffemodel_layerParam->blobs(bi).data(k));
                     }
                   }
       
               cv::Mat points(weights);
               cv::Mat labels,centers;
               cv::kmeans(points, cluster_count, labels, cv::TermCriteria( cv::TermCriteria::EPS+cv::TermCriteria::COUNT, 10, 1.0),
               3, flag, centers);
               
               /*
               BlobProto* model_cluster_index_table_proto = caffemodel_layerParam->mutable_quant_table();
	           model_cluster_index_table_proto->mutable_shape()->mutable_dim()->Add(cluster_count);
	           for (size_t k = 0; k < cluster_count; k++) {
		         model_cluster_index_table_proto->add_data(centers.at<float>(k));
	           }
               */
               
               size_t offset=0;
               for (size_t bi = 0; bi < caffemodel_layerParam->blobs_size(); bi++) {
                    //BlobProto* new_blob = caffemodel_layerParam->mutable_quant_blobs()->Add();
                    //*new_blob->mutable_shape() = caffemodel_layerParam->blobs(bi).shape();

                    size_t weight_mul = 1;
		    for (size_t i = 0; i < caffemodel_layerParam->blobs(bi).shape().dim_size(); i++) {
			   weight_mul *= caffemodel_layerParam->blobs(bi).shape().dim(i);
		    }
                    for (size_t k = 0; k < weight_mul; k++) {
                           caffemodel_layerParam->mutable_blobs(bi)->add_quant_data(labels.at<int>((int)(offset+k))); 
                    }
                    offset+=weight_mul;
                    for (size_t k = 0; k < cluster_count; k++) {
		         caffemodel_layerParam->mutable_blobs(bi)->add_quant_table(centers.at<float>(k));
	            }
                    caffemodel_layerParam->mutable_blobs(bi)->set_table_size(cluster_count);
                    caffemodel_layerParam->mutable_blobs(bi)->clear_data();
               }
               
	          //caffemodel_layerParam->clear_blobs();
                   
                  
                   caffemodel_tableInnerProductParam->set_table_size(cluster_count);
	           prototxt_tableInnerProductParam->set_table_size(cluster_count);

	           // Removing the source float values
	           caffemodel_layerParam->clear_inner_product_param();
	           prototxt_layerParam->clear_inner_product_param();

	           //caffemodel_layerParam->mutable_param()->Clear();
	           //prototxt_layerParam->mutable_param()->Clear();
        }
    }

    WriteProtoToTextFile(prototxt_net_param, prototxt_out_filename);
    WriteProtoToBinaryFile(caffemodel_net_param, caffemodel_out_filename); /*Binary*/

    LOG(INFO) << "Wrote upgraded NetParameter binary proto to " << caffemodel_out_filename;
    LOG(INFO) << "Wrote upgraded NetParameter text proto to " << prototxt_out_filename;

  return 0;
}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV

