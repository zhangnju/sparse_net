#!/usr/bin/env sh
set -e
rm -f examples/cifar10/log.info
TOOLS=./build/tools

$TOOLS/caffe train --solver=examples/cifar10/cifar10_quick_solver.prototxt --pruning=0 >> examples/cifar10/log.info 2>&1

# reduce learning rate by factor of 10 after 8 epochs
$TOOLS/caffe train --solver=examples/cifar10/cifar10_quick_solver_lr1.prototxt --snapshot=examples/cifar10/cifar10_quick_iter_4000.solverstate --pruning=0 >> examples/cifar10/log.info 2>&1
