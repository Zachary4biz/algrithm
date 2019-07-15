#!/usr/bin/env bash
########
# 参考自： https://www.tensorflow.org/hub/tutorials/image_retraining
# mkdir ~/example_code
# cd ~/example_code
# curl -LO https://github.com/tensorflow/hub/raw/master/examples/image_retraining/retrain.py
#
########
basicPath="/Users/zac/Downloads"
python retrain.py \
--bottleneck_dir="$basicPath/tf_files/bottlenecks" \
--how_many_training_steps 4000 \
--model_dir="$basicPath/tf_files/inception" \
--output_graph="$basicPath/tf_files/retrained_graph.pb" \
--output_labels="$basicPath/tf_files/retrained_labels.txt" \
--image_dir "/Users/zac/Downloads/enthnicity_img_copied/train"
