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

using std::ofstream;

using namespace caffe;  // NOLINT(build/namespaces)
using namespace std;

struct cluster {
    double center;
    std::vector<double> points;
    double min, max;
};

struct FoundClustersResult {
    map<size_t, size_t> indexRemap;
    vector<float> clusterCenters;
};

vector<cluster> init_clusters_hyperbolic(int cluster_count, double min, double max, vector<float> values) {
    // Splitting hyperbolic
    double x1 = 0.0001;
    double k = cluster_count / 2;
    double c = 1.0 - pow( x1 / max, 1.0 / (k-1) );

    vector<cluster> clusters;
    clusters.push_back(cluster());
    clusters.back().center = x1;
    clusters.push_back(cluster());
    clusters.back().center = -x1;

    for (int i = 0; i < k - 1; i++) {
	double xi = fabs(clusters.back().center) / (1.0 - c);
	clusters.push_back(cluster());
	clusters.back().center = xi;
	clusters.push_back(cluster());
	clusters.back().center = -xi;
    }
   
    return clusters;
}

vector<cluster> init_clusters_linear(int cluster_count, double min, double max, vector<float> values) {
    // Splitting linear
    double clusterWidth = (max - min) / cluster_count;
    vector<cluster> clusters;
    for (int i = 0; i < cluster_count; i++) {
	double xi = min + clusterWidth*i + clusterWidth / 2.0;
	clusters.push_back(cluster());
	clusters.back().center = xi;
    }
    
    return clusters;
}

FoundClustersResult find_clusters(vector<float> values, int cluster_count, string initType) {
    FoundClustersResult res;
    if (values.size() == 0) return res;
    if (cluster_count == 0) return res;

    // Searching for minimum and maximum
    double min = values[0], max = values[0];
    for (vector<float>::iterator iter = values.begin(); iter != values.end(); iter ++) {
	if ((*iter) < min) min = *iter;
	if ((*iter) > max) max = *iter;
    }

    vector<cluster> clusters;
	
    if(initType == "hyp"){
	   clusters = init_clusters_hyperbolic(cluster_count, min, max, values);
    } else if(initType == "lin"){
	   clusters = init_clusters_linear(cluster_count, min, max, values);
    } else {
	   LOG(ERROR) << "Wrong init type: "<<initType;
	   clusters = init_clusters_hyperbolic(cluster_count, min, max, values);
    }

    double delta;
    do {
	// Removing points from clusters
	for (vector<cluster>::iterator iter = clusters.begin(); iter != clusters.end(); iter ++) {
		iter->points.clear();
	}

	// Sorting points between clusters
	for (vector<float>::iterator val_iter = values.begin(); val_iter != values.end(); val_iter ++) {
                // Searching for the closest cluster
		double dist = fabs(*val_iter - clusters.begin()->center);
		vector<cluster>::iterator closest_cluster = clusters.begin();
		for (vector<cluster>::iterator iter = clusters.begin(); iter != clusters.end(); iter ++) {
			double new_dist = fabs(*val_iter - iter->center);
			if (new_dist < dist) {
				dist = new_dist;
				closest_cluster = iter;
			}
		}
		closest_cluster->points.push_back(*val_iter);
		res.indexRemap[val_iter - values.begin()] = closest_cluster - clusters.begin();
	}

	// Searching for clusters that contain no points
	for (vector<cluster>::iterator iter = clusters.begin(); iter != clusters.end(); iter ++) {
		if (iter->points.size() == 0) {
                // Searching for the most far point
		double max_distance = 0;
		vector<cluster>::iterator c_iter_found;
		vector<double>::iterator p_iter_found;
		for (vector<cluster>::iterator c_iter = clusters.begin(); c_iter != clusters.end(); c_iter ++) {
			if(c_iter->points.size()!=1){
				for (vector<double>::iterator p_iter = c_iter->points.begin(); p_iter != c_iter->points.end(); p_iter ++) {
					double distance = fabs(*p_iter - c_iter->center);
					if (distance > max_distance) {
						max_distance = distance;
						c_iter_found = c_iter;
						p_iter_found = p_iter;
					}
				}
			}
		}

		// Moving the point from the original cluster to the empty one
		double point = *p_iter_found;
		c_iter_found->points.erase(p_iter_found);
		iter->points.push_back(point);
		iter->center = point;
		res.indexRemap[p_iter_found - c_iter_found->points.begin()] = iter - clusters.begin();
		}
	}
	// Updating cluster centers and counting delta
	delta = 0.0;
	for (vector<cluster>::iterator iter = clusters.begin(); iter != clusters.end(); iter ++) {
		double sum = 0;
		for (vector<double>::iterator val_iter = iter->points.begin(); val_iter != iter->points.end(); val_iter ++) {
			sum += *val_iter;
		}

		double new_center = sum / iter->points.size();
                delta += fabs(new_center - iter->center);
                iter->center = new_center;
	}
     } while (delta < (max - min) / cluster_count / 100);

     for (vector<cluster>::iterator iter = clusters.begin(); iter != clusters.end(); iter ++) {
	res.clusterCenters.push_back(iter->center);
     }

     double mini_delta = 0.000001;
     double max_disp = 0;
     for (vector<cluster>::iterator iter = clusters.begin(); iter != clusters.end(); iter ++) {
	LOG(INFO) << "center: " << iter->center << "; ";
	if (iter->points.size() > 0) {
		iter->min = (double)(*(iter->points.begin())) - mini_delta;
		iter->max = (double)(*(iter->points.begin())) + mini_delta;
		for (vector<double>::iterator val_iter = iter->points.begin(); val_iter != iter->points.end(); val_iter ++) {
			if (*val_iter < iter->min) iter->min = *val_iter - mini_delta;
			if (*val_iter > iter->max) iter->max = *val_iter + mini_delta;
		}
		double disp = fabs(iter->max - iter->min) * 100;
		LOG(INFO) << "min: " << iter->min << ", max: " << iter->max << ", points count: " << iter->points.size() << ", dispersion: " << disp << "%" << endl;
		if (disp > max_disp) max_disp = disp;
	}
     }
     LOG(INFO) << "Maximum dispersion: " << max_disp << "%" << endl;

     return res;
}

int main(int argc, char** argv) {
     FLAGS_alsologtostderr = 1;  // Print output to stderr (while still logging)
     ::google::InitGoogleLogging(argv[0]);
     if (argc != 7) {
        LOG(ERROR) << "Usage: "
        << "convert_inner_product_to_table <prototxt_in> <caffemodel_in> <prototxt_out> <caffemodel_out> <hyp or lin> <cluster_count>";
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
		//continue;
	  }

	  LOG(INFO) << caffemodel_layer_name;
	  if (caffemodel_layerParam->type() == "InnerProduct") {
               const InnerProductParameter& innerProductParam = caffemodel_layerParam->inner_product_param();
               LOG(INFO) << "Found InnerProduct layer. Converting to TableInnerProduct";

	       caffemodel_layerParam->set_type("TableInnerProduct");
	       prototxt_layerParam->set_type("TableInnerProduct");

	       TableInnerProductParameter* caffemodel_tableInnerProductParam = caffemodel_layerParam->mutable_table_inner_product_param();
	       TableInnerProductParameter* prototxt_tableInnerProductParam = prototxt_layerParam->mutable_table_inner_product_param();
               caffemodel_tableInnerProductParam->set_num_output(innerProductParam.num_output());
	       prototxt_tableInnerProductParam->set_num_output(innerProductParam.num_output());

	       if (caffemodel_layerParam->blobs(0).shape().dim_size() == 0) {
		   caffemodel_layerParam->mutable_blobs(0)->mutable_shape()->add_dim(caffemodel_layerParam->blobs(0).height());
		   caffemodel_layerParam->mutable_blobs(0)->mutable_shape()->add_dim(caffemodel_layerParam->blobs(0).width());
	       }

	       if ( caffemodel_layerParam->blobs_size() > 1 && caffemodel_layerParam->blobs(1).shape().dim_size() == 0) {
		   caffemodel_layerParam->mutable_blobs(1)->mutable_shape()->add_dim(caffemodel_layerParam->blobs(1).width());
	       }

	       // Calculating full count of the weights
	       size_t weight_count_sum = 0;
	       for (size_t bi = 0; bi < caffemodel_layerParam->blobs_size(); bi++) {
		    size_t weight_mul = 1;
		    for (size_t i = 0; i < caffemodel_layerParam->blobs(bi).shape().dim_size(); i++) {
			weight_mul *= caffemodel_layerParam->blobs(bi).shape().dim(i);
		    }
		    weight_count_sum += weight_mul;
	       }

	       LOG(INFO) << "Full blob values count: " << weight_count_sum;

	       vector<float> index_table;
               // Filling in index table and integer matrix
	       int table_index = 0;
	       for (size_t bi = 0; bi < caffemodel_layerParam->blobs_size(); bi++) {
		    BlobProto* new_int_blob = caffemodel_layerParam->mutable_int_blobs()->Add();
                    *new_int_blob->mutable_shape() = caffemodel_layerParam->blobs(bi).shape();
                    size_t weight_mul = 1;
		    for (size_t i = 0; i < caffemodel_layerParam->blobs(bi).shape().dim_size(); i++) {
			weight_mul *= caffemodel_layerParam->blobs(bi).shape().dim(i);
		    }

		    for (size_t k = 0; k < weight_mul; k++) {
			// Putting the value into the table
                        index_table.push_back(caffemodel_layerParam->blobs(bi).data(k));
                        // Saving the table index into the int matrix
			new_int_blob->mutable_int_data()->Add(table_index);
			// Incrementing the table index
			table_index ++;
		    }
		}

		// Clearing floating blobs and adding the index blob
		caffemodel_layerParam->clear_blobs();

                FoundClustersResult fcr;
		fcr = find_clusters(index_table, cluster_count, initType);

		BlobProto* model_cluster_index_table_proto = caffemodel_layerParam->mutable_blobs()->Add();
		model_cluster_index_table_proto->mutable_shape()->mutable_dim()->Add(cluster_count);

		for (size_t k = 0; k < cluster_count; k++) {
			model_cluster_index_table_proto->add_data(fcr.clusterCenters[k]);
		}

		for (size_t bi = 0; bi < caffemodel_layerParam->int_blobs_size(); bi++) {
			size_t weight_mul = 1;
			for (size_t i = 0; i < caffemodel_layerParam->int_blobs(bi).shape().dim_size(); i++) {
				weight_mul *= caffemodel_layerParam->int_blobs(bi).shape().dim(i);
			}

			for (size_t k = 0; k < weight_mul; k++) {
				// Saving the id number of the cluster contained the current weight into the int matrix
				size_t cur_int_data = caffemodel_layerParam->int_blobs(bi).int_data(k);
				size_t cur_cluster_id = fcr.indexRemap[cur_int_data];
				caffemodel_layerParam->mutable_int_blobs(bi)->set_int_data(k, cur_cluster_id);
			}
		 }

		 // Allocating the index table
		 caffemodel_tableInnerProductParam->set_table_size(cluster_count);
		 prototxt_tableInnerProductParam->set_table_size(cluster_count);

		 // Removing the source float values
		 caffemodel_layerParam->clear_inner_product_param();
		 prototxt_layerParam->clear_inner_product_param();

		 caffemodel_layerParam->mutable_param()->Clear();
		 prototxt_layerParam->mutable_param()->Clear();
        }
    }

    WriteProtoToTextFile(prototxt_net_param, prototxt_out_filename);
    WriteProtoToBinaryFile(caffemodel_net_param, caffemodel_out_filename); /*Binary*/

    LOG(INFO) << "Wrote upgraded NetParameter binary proto to " << caffemodel_out_filename;
    LOG(INFO) << "Wrote upgraded NetParameter text proto to " << prototxt_out_filename;
  
    return 0;
}
