#!/usr/bin/env sh

CAFFE_HOME=`pwd`

SOLVER=examples/yolo2/solver.prototxt
WEIGHTS=$1
$CAFFE_HOME/build/tools/caffe train --solver=$SOLVER --weights=$WEIGHTS
