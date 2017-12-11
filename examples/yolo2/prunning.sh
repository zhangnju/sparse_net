#!/usr/bin/env sh

CAFFE_HOME=`pwd`

current_time=$(date | sed s/[[:space:]]//g)
#current_time=${current_time// /_}
#current_time=${current_time//:/-}

snapshot_path=$CAFFE_HOME/examples/yolo2/${current_time}
mkdir $snapshot_path

SOLVER=examples/yolo2/finetune_solver.prototxt
WEIGHTS=$1
$CAFFE_HOME/build/tools/caffe train --solver=$SOLVER --weights=$WEIGHTS > "${snapshot_path}/train.info" 2>&1
cat ${snapshot_path}/train.info | grep "loss ="| awk '{print $5, $6 , $11 $12 $13}'>${snapshot_path}/loss.info
