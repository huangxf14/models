#!/bin/bash


model_name="mobilenet_v2_035_sgmt"
log_dir="dec1/train"

#dataset_name="coco2017_saliency_ext"
#tfrecord_dir="coco2017/saliency_ext/tfrecord"
#dataset_name="owlii_studio"
#tfrecord_dir="owlii_studio/tfrecord"
dataset_name="LV-instance-simple"
tfrecord_dir="LV-instance1/tfrecord_simple"

ckpt="model.ckpt-250000"
#checkpoint_exclude_scopes="MobilenetV2/Logits,MobilenetV2/Conv_1,MobilenetV2/Conv_2,MobilenetV2/Conv_3,MobilenetV2/Conv_4,MobilenetV2/Decoder"
trainable_scopes="MobilenetV2/Logits,MobilenetV2/Conv_2,MobilenetV2/Conv_2,MobilenetV2/Conv_3,MobilenetV2/Decoder"

num_clones_new=1
batch_size_new=40    # 192 * 2  
train_steps_new=10000  # 10000 steps, about 8 epoches
second_stage_dir="all"
num_clones=1
batch_size=20
train_steps=400000     # 100000 steps, about 60 epoches
lr_decay_factor=0.87

###########################################
HOME="/home/corp.owlii.com/xiufeng.huang"
SLIM="${HOME}/models/research/slim"
WORKSPACE="${HOME}/models/workspace/seg"
DATASET_DIR="${HOME}/models/workspace/seg/${tfrecord_dir}"
INIT_CHECKPOINT="${HOME}/models/workspace/seg/coco-instance/model035_256/all/${ckpt}"
TRAIN_DIR="${HOME}/models/workspace/seg/LV-instance1/model035_256"
PYTHONPATH="${PYTHONPATH}:${WORKSPACE}"
#mkdir -p ${TRAIN_DIR}

##### Start training #####
# Fine-tune only the new layers
python train_sgmt.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=${dataset_name} \
  --dataset_split_name=LV-train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${model_name} \
  --checkpoint_path=${INIT_CHECKPOINT} \
  --checkpoint_exclude_scopes=${checkpoint_exclude_scopes} \
  --trainable_scopes=${trainable_scopes} \
  --max_number_of_steps=${train_steps_new} \
  --batch_size=${batch_size_new} \
  --min_scale_factor=0.5 \
  --max_scale_factor=2.0 \
  --scale_factor_step_size=0 \
  --learning_rate=0.01 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=600 \
  --save_summaries_secs=600 \
  --max_ckpts_to_keep=3 \
  --keep_ckpt_every_n_hours=1 \
  --log_every_n_steps=10 \
  --optimizer=rmsprop \
  --weight_decay=0.00004 \
  --num_clones=${num_clones_new} \
  --use_decoder=True \
  --instance_seg=True \
  --inner_extension_ratio=-0.1 \
  --outer_extension_ratio=0.2 \
  --filling=central_padding  # central_padding, resize


# Run evaluation.
python eval_sgmt.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=${dataset_name} \
  --dataset_split_name=LV-test \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${model_name} \
  --batch_size=8 \
  --min_resize_value=512 \
  --max_resize_value=512 \
  --use_decoder=True \
  --max_number_of_evaluations 1 \
  --instance_seg=True \
  --inner_extension_ratio=-0.1 \
  --outer_extension_ratio=0.2 \
  --filling=central_padding  # central_padding, resize


# Fine-tune all the layers
python train_sgmt.py \
  --train_dir=${TRAIN_DIR}/${second_stage_dir} \
  --dataset_name=${dataset_name} \
  --dataset_split_name=LV-train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${model_name} \
  --checkpoint_path=${TRAIN_DIR} \
  --max_number_of_steps=${train_steps} \
  --batch_size=${batch_size} \
  --min_scale_factor=0.5 \
  --max_scale_factor=2.0 \
  --scale_factor_step_size=0 \
  --learning_rate=0.01 \
  --learning_rate_decay_type=exponential \
  --learning_rate_decay_factor=${lr_decay_factor} \
  --save_interval_secs=1200 \
  --save_summaries_secs=600 \
  --max_ckpts_to_keep=3 \
  --keep_ckpt_every_n_hours=2.0 \
  --log_every_n_steps=100 \
  --optimizer=rmsprop \
  --weight_decay=0.00004 \
  --num_clones=${num_clones} \
  --use_decoder=True \
  --instance_seg=True \
  --inner_extension_ratio=-0.1 \
  --outer_extension_ratio=0.2 \
  --filling=central_padding  # central_padding, resize

# Run evaluation.
python eval_sgmt.py \
  --checkpoint_path=${TRAIN_DIR}/${second_stage_dir} \
  --eval_dir=${TRAIN_DIR}/${second_stage_dir} \
  --dataset_name=${dataset_name} \
  --dataset_split_name=LV-test \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${model_name} \
  --batch_size=8 \
  --min_resize_value=512 \
  --max_resize_value=512 \
  --use_decoder=True \
  --max_number_of_evaluations 1 \
  --instance_seg=True \
  --inner_extension_ratio=-0.1 \
  --outer_extension_ratio=0.2 \
  --filling=resize  # central_padding, resize



