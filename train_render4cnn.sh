#!/usr/bin/env sh

./build/tools/caffe train  --solver=models/train/solver_ohem.prototxt \
--weights=models/train/finetune_voc_2012_train_iter_70k.caffemodel
