#!/usr/bin/env sh
set -e

TOOLS=./build/tools

$TOOLS/caffe test \
  --model=examples/googlenet/train_val_sp.prototxt --weights=/home/zhangning/source_code/caffe_model/bvlc_googlenet.caffemodel -gpu=0 $@

