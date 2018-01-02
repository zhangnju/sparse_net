#!/usr/bin/env sh
set -e
rm -f examples/cifar10/log.info
TOOLS=./build/tools

$TOOLS/caffe train --solver=examples/cifar10/cifar10_quick_ft_solver.prototxt --weights=examples/cifar10/cifar10_quick_origin.caffemodel --pruning=1 >> examples/cifar10/log.info 2>&1

