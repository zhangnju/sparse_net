#!/usr/bin/env sh
set -e

TOOLS=./build/tools

$TOOLS/caffe test \
  -model=examples/cifar10/cifar10_quick_train_test.prototxt -weights=examples/cifar10/cifar10_quick_iter_4000.caffemodel -gpu=0 $@

