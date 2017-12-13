__author__ = 'pittnuts'
import caffe
import re
from pittnuts import *
import os
import matplotlib.pyplot as plt
import argparse
import caffeparser
import matplotlib.cm as cm

def show_filters(net,layername):
    weights = net.params[layername][0].data
    if len(weights.shape) < 3:
        return
    weight_scope = abs(weights).max()
    filt_min = -weight_scope
    filt_max = weight_scope

    chan_num = weights.shape[1]
    display_region_size = ceil(sqrt(chan_num))
    for n in range(min(1000,weights.shape[0])):
        if sum(abs(weights[n]))>0:
            print "{}-th channel is usefull".format(n)
            plt.figure()
            for c in range(chan_num):
                plt.subplot((int)(display_region_size),(int)(display_region_size),c+1)
                if sum(abs(weights[n,c]))>0:
                    #plt.title("filter #{} output".format(c))
                    plt.imshow(weights[n,c],vmin=filt_min,vmax=filt_max,cmap=plt.get_cmap('seismic'),interpolation='none')
                    #plt.tight_layout()
                plt.tick_params(which='both',labelbottom='off',labelleft='off',bottom='off',top='off',left='off',right='off')


if __name__ == "__main__":
    # helper show filter outputs
    parser = argparse.ArgumentParser()
    parser.add_argument('--prototxt', type=str, required=True)
    parser.add_argument('--origimodel', type=str, required=True)
    args = parser.parse_args()
    prototxt = args.prototxt 
    original_caffemodel = args.origimodel 
    net_parser = caffeparser.CaffeProtoParser(prototxt)
    net_msg = net_parser.readProtoNetFile()

    caffe.set_mode_cpu()
    # GPU mode
    #caffe.set_device(1)
    #caffe.set_mode_gpu()
    orig_net = caffe.Net(prototxt,original_caffemodel, caffe.TEST)
    print("blobs {}\nparams {}".format(orig_net.blobs.keys(), orig_net.params.keys()))

    r_width = 0.0001
    #einet_plt = plt.figure().add_subplot(111)
    for layer_name in orig_net.params.keys():
            layer_type = net_parser.getLayerByName(net_msg,layer_name).type
            if layer_type=='Convolution' or layer_type =='InnerProduct':
                print "analyzing {}".format(layer_name)
                weights_orig = orig_net.params[layer_name][0].data
                print "[{}] original: %{} zeros".format(layer_name,100*sum((abs(weights_orig)<r_width).flatten())/(float)(weights_orig.size))
                
