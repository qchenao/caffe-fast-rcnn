#!/usr/bin/env sh

./build/tools/caffe train  --solver=models/train/solver_baseline.prototxt \
--weights=models/train/finetune_voc_2012_train_iter_70k.caffemodel -gpu 1
