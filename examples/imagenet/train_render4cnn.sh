#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=models/train/solver_ohem.prototxt
    --gpu 1
