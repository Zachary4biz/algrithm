#!/usr/bin/env bash
#################################################
# tensorflow 官方例子
# predict.py和predict.sh来自Siraj Raval 的git项目
##################################################
imgPath=$1
basicPath="/Users/zac/Downloads"
python label_image.py \
--graph="$basicPath/tf_files/retrained_graph.pb" \
--labels="$basicPath/tf_files/retrained_labels.txt" \
--input_layer=Placeholder \
--output_layer=final_result \
--image=$imgPath
