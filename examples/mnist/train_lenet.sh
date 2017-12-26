#!/usr/bin/env sh
set -e
rm -f examples/mnist/log.info
./build/tools/caffe train --solver=examples/mnist/lenet_solver.prototxt >> examples/mnist/log.info 2>&1
