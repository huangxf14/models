#!/bin/bash


log_dir="coco-instance/model075_resize/all"
model_name="mobilenet_v2_075_sgmt"

#dataset_name="coco2017_saliency_ext"
#tfrecord_dir="coco2017/saliency_ext/tfrecord"
#dataset_name="owlii_studio"
#tfrecord_dir="owlii_studio/tfrecord"
dataset_name="coco-instance-simple"
tfrecord_dir="coco-instance/tfrecord_simple"

###################################
HOME="/home/corp.owlii.com/xiufeng.huang"
SLIM="${HOME}/models/research/slim"
WORKSPACE="${HOME}/models/workspace/seg"
DATASET_DIR="${HOME}/models/workspace/seg/${tfrecord_dir}"
VAL_DIR="${WORKSPACE}/${log_dir}"
PYTHONPATH="${PYTHONPATH}:${WORKSPACE}"   
# Run evaluation.
python eval_sgmt.py \
  --checkpoint_path=${VAL_DIR} \
  --eval_dir=${VAL_DIR} \
  --dataset_name=${dataset_name} \
  --dataset_split_name=coco_test_instance \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${model_name} \
  --batch_size=1 \
  --eval_interval_secs=10 \
  --use_cpu=True \
  --min_resize_value=512 \
  --max_resize_value=512 \
  --use_decoder=True \
  --max_number_of_evaluations 0 \
  --instance_seg=True \
  --inner_extension_ratio=-0.1 \
  --outer_extension_ratio=0.2 \
  --filling=resize  # central_padding, resize

