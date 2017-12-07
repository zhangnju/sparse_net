__author__ = 'pittnuts'
import caffe
import re
from pittnuts import *
import os
import matplotlib.pyplot as plt
import argparse
import caffeparser
import matplotlib.cm as cm

if __name__ == "__main__":
    # helper show filter outputs
    parser = argparse.ArgumentParser()
    parser.add_argument('--prototxt', type=str, required=True)
    parser.add_argument('--origimodel', type=str, required=True)
    args = parser.parse_args()
    prototxt = args.prototxt #"models/eilab_reference_sparsenet/train_val_scnn.prototxt"
    original_caffemodel = args.origimodel # "models/eilab_reference_sparsenet/eilab_reference_sparsenet.caffemodel"
    net_parser = caffeparser.CaffeProtoParser(prototxt)
    net_msg = net_parser.readProtoNetFile()

    caffe.set_mode_cpu()
    # GPU mode
    #caffe.set_device(1)
    #caffe.set_mode_gpu()
    orig_net = caffe.Net(prototxt,original_caffemodel, caffe.TEST)
    print("blobs {}\nparams {}".format(orig_net.blobs.keys(), orig_net.params.keys()))
    kernel_max_sizexsize = -1
    speedupinfo = ""
    plot_count = 0
    subplot_num = 0
    for layer_name in orig_net.params.keys():
        layer_type = net_parser.getLayerByName(net_msg,layer_name).type
        if layer_type =='Convolution':
            subplot_num += net_parser.getLayerByName(net_msg,layer_name).convolution_param.group
        elif layer_type =='InnerProduct':
            subplot_num += 1

    zero_threshold = 0.0001
    for layer_name in orig_net.params.keys():
            layer_type = net_parser.getLayerByName(net_msg,layer_name).type
            if layer_type=='Convolution' or layer_type =='InnerProduct':
                print "analyzing {}".format(layer_name)
                weights_orig = orig_net.params[layer_name][0].data
                print "[{}] original: %{} zeros".format(layer_name,100*sum((abs(weights_orig)<zero_threshold).flatten())/(float)(weights_orig.size))
            