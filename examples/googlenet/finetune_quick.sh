#!/usr/bin/env sh
set -e
rm -f examples/googlenet/log.info
TOOLS=./build/tools

$TOOLS/caffe train --solver=examples/googlenet/ft_quick_solver.prototxt --weights=~/source_code/caffe_model/bvlc_googlenet.caffemodel --pruning=1 >> examples/googlenet/log.info 2>&1

