__author__ = 'pittnuts'
import caffe
import re
from pittnuts import *
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import caffeparser
def print_eig_info(eig_values,percent=0.95):
    eig_sum = sum(eig_values)
    #print eig_values
    for i in range(1, eig_values.size):
        eig_values[i] = eig_values[i] + eig_values[i - 1]
    eig_values = eig_values / eig_sum
    for i in range(1, eig_values.size):
        if eig_values[i]>percent:
            print "{} / {} is more than {} of eigenvalue sum".format(i+1,eig_values.size,percent)
            break

def show_filters(net,layername ,filt_min ,filt_max):
    rgb = False
    weights = net.params[layername][0].data
    if len(weights.shape) < 3:
        return
    chan_num = weights.shape[1]
    filter_num = weights.shape[0]
    #display_region_size = ceil(sqrt(filter_num))
    rgb = (chan_num==3)
    plt.figure()
    if rgb:
        for n in range(filter_num):
            plt.subplot(6, 16,  n + 1)
            img = (weights[n, :].transpose((1,2,0)) - filt_min)/(filt_max-filt_min)
            plt.imshow(img,  interpolation='none')
            plt.tick_params(which='both', labelbottom='off', labelleft='off', bottom='off', top='off', left='off', right='off')
    else:
        #c_ordered=(3,10,12,0,1,2,4,5,6,7,8,9,11,13,14,15,16,17,18,19)
        for c in range(min(20,chan_num)):
            #if sum(abs(weights[:,c,:,:]))>0:
                #print "{}-th channel is usefull".format(c)
                for n in range(filter_num):
                    #plt.subplot((int)(display_region_size),(int)(display_region_size),n+1)
                    plt.subplot(chan_num,filter_num, filter_num*c + n + 1)
                    #if sum(abs(weights[n,c]))>0:
                    plt.imshow(weights[n,c],vmin=filt_min,vmax=filt_max,cmap=plt.get_cmap('Greys'),interpolation='none')
                    #plt.imshow(weights[n,c_ordered[c]], vmin=filt_min, vmax=filt_max, cmap=plt.get_cmap('Greys'), interpolation='none')
                    plt.tick_params(which='both',labelbottom='off',labelleft='off',bottom='off',top='off',left='off',right='off')

def show_2Dfilter_pca(net,layername,showit=False):
    weights = net.params[layername][0].data
    if len(weights.shape) < 3:
        return
    filter_num = weights.shape[0]
    chan_num = weights.shape[1]
    kernel_h = weights.shape[2]
    kernel_w = weights.shape[3]
    kernel_size = kernel_h*kernel_w

    weights_pca = weights.reshape((chan_num*filter_num, kernel_size)).transpose()
    weights_pca, eig_vecs, eig_values = pca(weights_pca)
    print_eig_info(eig_values)
    if showit:
        weights_pca = weights_pca.transpose().reshape(filter_num,chan_num,kernel_h,kernel_w)
        filt_max = abs(weights_pca).max()
        filt_min = -filt_max
        #eig_vecs = eig_vecs.transpose().reshape(kernel_size,kernel_h,kernel_w)
        plt.figure()
        for c in range(min(20, chan_num)):
            for n in range(filter_num):
                plt.subplot(chan_num, filter_num, filter_num * c + n + 1)
                plt.imshow(weights_pca[n, c], vmin=filt_min, vmax=filt_max, cmap=plt.get_cmap('Greys'), interpolation='none')
                plt.tick_params(which='both', labelbottom='off', labelleft='off', bottom='off', top='off', left='off',right='off')

def show_filter_channel_pca(net,layername):
    weights = net.params[layername][0].data
    if len(weights.shape) < 3:
        return
    filter_num = weights.shape[0]
    chan_num = weights.shape[1]
    kernel_h = weights.shape[2]
    kernel_w = weights.shape[3]
    kernel_size = kernel_h*kernel_w

    # filter-wise
    print layername+" analyzing filter-wise:"
    weights_pca = weights.reshape((filter_num, chan_num*kernel_size)).transpose()
    weights_pca, eig_vecs, eig_values = pca(weights_pca)
    print_eig_info(eig_values)
    weights_pca = weights_pca.transpose().reshape(filter_num,chan_num,kernel_h,kernel_w)
    # channel-wise
    print layername+" analyzing channel-wise:"
    weights_pca = weights_pca.transpose((1,0,2,3)).reshape((chan_num,  filter_num* kernel_size)).transpose()
    weights_pca, eig_vecs, eig_values = pca(weights_pca)
    print_eig_info(eig_values)

    #weights_pca = weights_pca.transpose().reshape(chan_num, filter_num, kernel_h, kernel_w).transpose((1,0,2,3))
    #filt_max = abs(weights_pca).max()
    #filt_min = -filt_max
    ##eig_vecs = eig_vecs.transpose().reshape(kernel_size,kernel_h,kernel_w)
    #plt.figure()
    #for c in range(min(20, chan_num)):
    #    for n in range(filter_num):
    #        plt.subplot(chan_num, filter_num, filter_num * c + n + 1)
    #        plt.imshow(weights_pca[n, c], vmin=filt_min, vmax=filt_max, cmap=plt.get_cmap('Greys'), interpolation='none')
    #        plt.tick_params(which='both', labelbottom='off', labelleft='off', bottom='off', top='off', left='off',right='off')

def show_filter_shapes(net, layername):
    weights = net.params[layername][0].data
    if len(weights.shape) < 3:
        return
    chan_num = weights.shape[1]
    filter_num = weights.shape[0]
    weights = abs(weights)
    weights = sum(weights,axis=0)!=0
    plt.figure()
    for c in range(min(20, chan_num)):
        plt.subplot(chan_num, 1, c + 1)
        plt.imshow(weights[c], vmin=0, vmax=1, cmap=plt.get_cmap('Greys'), interpolation='none')
        plt.tick_params(which='both', labelbottom='off', labelleft='off', bottom='off', top='off', left='off', right='off')

# get the maximum abs weight
def get_max_weight(orig_net,layer_name):
    weight_scope = 0
    weights_orig = orig_net.params[layer_name][0].data
    max_val = abs(weights_orig).max()
    if max_val > weight_scope:
        weight_scope = max_val
    return weight_scope

# get the min abs weight
def get_min_weight(orig_net,layer_name):
    weights_orig = orig_net.params[layer_name][0].data
    min_val = abs(weights_orig).min()
    return min_val

def drawHist(orig_net,layer_name):
    weights= orig_net.params[layer_name][0].data
    filter_num = weights.shape[0]
    chan_num = weights.shape[1]
    kernel_h = weights.shape[2]
    kernel_w = weights.shape[3]
    kernel_size = kernel_h*kernel_w
    new_weights=weights.reshape((chan_num*filter_num, kernel_size)).flatten()
    plt.hist(new_weights,100)
    plt.xlabel('Weight')
    plt.ylabel('Frequnecy')
    plt.title("analyzing {}".format(layer_name))
    plt.show()

def get_threashold(orig_net,layer_name,zero_rate):
    weights= orig_net.params[layer_name][0].data
    if 4==len(weights.shape):
      filter_num = weights.shape[0]
      chan_num = weights.shape[1]
      kernel_h = weights.shape[2]
      kernel_w = weights.shape[3]
      kernel_size = kernel_h*kernel_w
      weight_array=weights.reshape((chan_num*filter_num, kernel_size)).flatten()
    else:
      weight_array=weights.reshape((1, weights.shape[0]*weights.shape[1])).flatten()
    sorted_weight=np.argsort(abs(weight_array))
    ind=sorted_weight[int(zero_rate*len(weight_array))]
    
    return abs(weight_array[ind])

def rewrite(orig_prototxt,thres=[]):
    new_prototxt=orig_prototxt.replace('.prototxt','_sp.prototxt')
    if os.path.exists(new_prototxt):
       os.remove(new_prototxt)
    oldfile=open(orig_prototxt,"r")
    newfile=open(new_prototxt,"w")
    index=0
    for line in oldfile:
       if line.find("convolution_param")!=-1:
           newfile.write("  pruning_thres:{}\n".format(thres[index]));
           newfile.write(line)
           index+=1
       elif line.find("inner_product_param")!=-1:
           newfile.write("  pruning_thres:{}\n".format(thres[index]));
           newfile.write(line)
           index+=1
       else:
           newfile.write(line)
       
    oldfile.close()
    newfile.close()

if __name__ == "__main__":
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
    thres_list=[]
    for layer_name in orig_net.params.keys():
       layer_type = net_parser.getLayerByName(net_msg,layer_name).type
       if layer_type=='Convolution' or layer_type =='InnerProduct':
            print "analyzing {}".format(layer_name)
            max_val=get_max_weight(orig_net,layer_name)
            print "the max weight in this layer is {}".format(max_val)
            min_val=get_min_weight(orig_net,layer_name)
            print "the min weight in this layer is {}".format(min_val)
            thres=get_threashold(orig_net,layer_name,0.8)
            print "the pruning threashold in this layer is {}".format(thres) 
            thres_list.append(thres)
    
    rewrite(prototxt,thres_list)
