#!/bin/bash


log_dir="RGMP/modelvideoseq/ten"
model_name="mobilenet_v2_035_video_trainseq"

#dataset_name="coco2017_saliency_ext"
#tfrecord_dir="coco2017/saliency_ext/tfrecord"
#dataset_name="owlii_studio"
#tfrecord_dir="owlii_studio/tfrecord"
dataset_name="DAVIS-test"
tfrecord_dir="RGMP/tfrecord-test"

###################################
HOME="/home/corp.owlii.com/xiufeng.huang"
SLIM="${HOME}/models/research/slim"
WORKSPACE="${HOME}/models/workspace/seg"
DATASET_DIR="${HOME}/models/workspace/seg/${tfrecord_dir}"
VAL_DIR="${WORKSPACE}/${log_dir}"
PYTHONNPATH="${PYTHONPATH}:${WORKSPACE}"


# Run evaluation.
python eval_sgmt.py \
  --checkpoint_path=${VAL_DIR} \
  --eval_dir=${VAL_DIR} \
  --dataset_name=${dataset_name} \
  --dataset_split_name=test \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${model_name} \
  --batch_size=1 \
  --use_cpu=True \
  --eval_interval_secs=10 \
  --min_resize_value=512 \
  --max_resize_value=512 \
  --train_list=False \
  --use_decoder=True \
  --max_number_of_evaluations 0  # 0 for infinite loop
#  --instance_seg=True \
#  --inner_extension_ratio=-0.1 \
#  --outer_extension_ratio=0.2 \
#  --filling=resize  # central_padding, resize


