#!/bin/bash


model_name="mobilenet_v2_1_video_final"
log_dir="dec1/train"

#dataset_name="coco2017_saliency_ext"
#tfrecord_dir="coco2017/saliency_ext/tfrecord"
#dataset_name="owlii_studio"
#tfrecord_dir="owlii_studio/tfrecord"
dataset_coco_name="COCO-trans"
tfrecord_coco_dir="RGMP/tfrecord-cocotrans"
dataset_init_name="DAVIS"
tfrecord_init_dir="RGMP/tfrecord"
dataset_name="DAVIS-video10"
tfrecord_dir="RGMP/tfrecord-video10"

ckpt="model.ckpt-400000"
checkpoint_exclude_scopes="MobilenetV2-Decoder/Logits,MobilenetV2/Conv,MobilenetV2/Conv_1,MobilenetV2-Decoder/GC,MobilenetV2-Decoder/GC_Residual,MobilenetV2-Decoder/Decoder"
trainable_scopes="MobilenetV2-Decoder/Logits,MobilenetV2/Conv,MobilenetV2/Conv_1,MobilenetV2-Decoder/GC,MobilenetV2-Decoder/GC_Residual,MobilenetV2-Decoder/Decoder"

coco_stage_dir="trans"
train_steps_coco=200000

num_clones_new=1
batch_size_new=15    # 192 * 2  
train_steps_new=5000  # 10000 steps, about 8 epoches
second_stage_dir="all"
num_clones=1
batch_size=10
train_steps=20000     # 100000 steps, about 60 epoches

batch_two_size=5
train_two_steps=20000
two_stage_dir="two"
batch_four_size=2
train_four_steps=20000
four_stage_dir="four"
batch_seven_size=1
train_seven_steps=20000
seven_stage_dir="seven"
batch_ten_size=1
train_ten_steps=200000
ten_stage_dir='ten'

lr_decay_factor=0.65

###########################################
HOME="/home/corp.owlii.com/xiufeng.huang"
SLIM="${HOME}/models/research/slim"
WORKSPACE="${HOME}/models/workspace/seg"
DATASET_INIT_DIR="${HOME}/models/workspace/seg/${tfrecord_init_dir}"
DATASET_COCO_DIR="${HOME}/models/workspace/seg/${tfrecord_coco_dir}"
DATASET_DIR="${HOME}/models/workspace/seg/${tfrecord_dir}"
INIT_CHECKPOINT="${HOME}/models/workspace/seg/coco-instance/model_1.0/all/${ckpt}"
TRAIN_DIR="${HOME}/models/workspace/seg/RGMP/modelvideofinal1"
#mkdir -p ${TRAIN_DIR}

##### Start training #####
# Fine-tune only the new layers
python train_video_sgmt.py \
  --train_dir=${TRAIN_DIR}/${coco_stage_dir} \
  --dataset_name=${dataset_coco_name} \
  --dataset_split_name=cocotrans_train_instance \
  --dataset_dir=${DATASET_COCO_DIR} \
  --model_name=${model_name} \
  --checkpoint_path=${INIT_CHECKPOINT} \
  --checkpoint_exclude_scopes=${checkpoint_exclude_scopes} \
  --trainable_scopes=${trainable_scopes} \
  --max_number_of_steps=${train_steps_new} \
  --batch_size=${batch_size_new} \
  --min_scale_factor=0.5 \
  --max_scale_factor=2.0 \
  --scale_factor_step_size=0 \
  --learning_rate=0.05 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=600 \
  --save_summaries_secs=600 \
  --max_ckpts_to_keep=3 \
  --keep_ckpt_every_n_hours=1 \
  --log_every_n_steps=10 \
  --optimizer=rmsprop \
  --weight_decay=0.00004 \
  --num_clones=${num_clones_new} \
  --use_decoder=True

# Run evaluation.
python eval_sgmt.py \
  --checkpoint_path=${TRAIN_DIR}/${coco_stage_dir} \
  --eval_dir=${TRAIN_DIR}/${coco_stage_dir}/${second_stage_dir} \
  --dataset_name=${dataset_coco_name} \
  --dataset_split_name=cocotrans_test_instance \
  --dataset_dir=${DATASET_COCO_DIR} \
  --model_name=${model_name} \
  --batch_size=8 \
  --min_resize_value=512 \
  --max_resize_value=512 \
  --train_list=False \
  --use_decoder=True \
  --max_number_of_evaluations 1  # 0 for infinite loop


# Fine-tune all the layers
python train_video_sgmt.py \
  --train_dir=${TRAIN_DIR}/${coco_stage_dir}/${second_stage_dir} \
  --dataset_name=${dataset_coco_name} \
  --dataset_split_name=cocotrans_train_instance \
  --dataset_dir=${DATASET_COCO_DIR} \
  --model_name=${model_name} \
  --checkpoint_path=${TRAIN_DIR}/${coco_stage_dir} \
  --max_number_of_steps=${train_steps_coco} \
  --batch_size=${batch_size} \
  --min_scale_factor=0.5 \
  --max_scale_factor=2.0 \
  --scale_factor_step_size=0 \
  --learning_rate=0.05 \
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
  --use_decoder=True

# Run evaluation.
python eval_sgmt.py \
  --checkpoint_path=${TRAIN_DIR}/${coco_stage_dir}/${second_stage_dir} \
  --eval_dir=${TRAIN_DIR}/${coco_stage_dir}/${second_stage_dir} \
  --dataset_name=${dataset_coco_name} \
  --dataset_split_name=cocotrans_test_instance \
  --dataset_dir=${DATASET_COCO_DIR} \
  --model_name=${model_name} \
  --batch_size=8 \
  --min_resize_value=512 \
  --max_resize_value=512 \
  --train_list=False \
  --use_decoder=True \
  --max_number_of_evaluations 1  # 0 for infinite loop


# Fine-tune all the layers
python train_video_sgmt.py \
  --train_dir=${TRAIN_DIR}/${second_stage_dir} \
  --dataset_name=${dataset_init_name} \
  --dataset_split_name=train-human \
  --dataset_dir=${DATASET_INIT_DIR} \
  --model_name=${model_name} \
  --checkpoint_path=${TRAIN_DIR}/${coco_stage_dir}/${second_stage_dir} \
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
  --use_decoder=True

# Run evaluation.
python eval_sgmt.py \
  --checkpoint_path=${TRAIN_DIR}/${second_stage_dir} \
  --eval_dir=${TRAIN_DIR}/${second_stage_dir} \
  --dataset_name=${dataset_init_name} \
  --dataset_split_name=val-human-2016 \
  --dataset_dir=${DATASET_INIT_DIR} \
  --model_name=${model_name} \
  --batch_size=8 \
  --min_resize_value=512 \
  --max_resize_value=512 \
  --train_list=False \
  --use_decoder=True \
  --max_number_of_evaluations 1  # 0 for infinite loop

python train_video_sgmt.py \
  --train_dir=${TRAIN_DIR}/${two_stage_dir} \
  --dataset_name=${dataset_name} \
  --dataset_split_name=train-human \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${model_name} \
  --checkpoint_path=${TRAIN_DIR}/${second_stage_dir} \
  --max_number_of_steps=${train_two_steps} \
  --batch_size=${batch_two_size} \
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
  --train_length=2 \
  --num_clones=${num_clones} \
  --use_decoder=True

python eval_sgmt.py \
  --checkpoint_path=${TRAIN_DIR}/${two_stage_dir} \
  --eval_dir=${TRAIN_DIR}/${second_stage_dir} \
  --dataset_name=${dataset_init_name} \
  --dataset_split_name=val-human-2016 \
  --dataset_dir=${DATASET_INIT_DIR} \
  --model_name=${model_name} \
  --batch_size=8 \
  --min_resize_value=512 \
  --max_resize_value=512 \
  --train_list=False \
  --use_decoder=True \
  --max_number_of_evaluations 1  # 0 for infinite loop


python train_video_sgmt.py \
  --train_dir=${TRAIN_DIR}/${four_stage_dir} \
  --dataset_name=${dataset_name} \
  --dataset_split_name=train-human \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${model_name} \
  --checkpoint_path=${TRAIN_DIR}/${two_stage_dir} \
  --max_number_of_steps=${train_four_steps} \
  --batch_size=${batch_four_size} \
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
  --train_length=4 \
  --num_clones=${num_clones} \
  --use_decoder=True

python eval_sgmt.py \
  --checkpoint_path=${TRAIN_DIR}/${four_stage_dir} \
  --eval_dir=${TRAIN_DIR}/${second_stage_dir} \
  --dataset_name=${dataset_init_name} \
  --dataset_split_name=val-human-2016 \
  --dataset_dir=${DATASET_INIT_DIR} \
  --model_name=${model_name} \
  --batch_size=8 \
  --min_resize_value=512 \
  --max_resize_value=512 \
  --train_list=False \
  --use_decoder=True \
  --max_number_of_evaluations 1  # 0 for infinite loop


python train_video_sgmt.py \
  --train_dir=${TRAIN_DIR}/${seven_stage_dir} \
  --dataset_name=${dataset_name} \
  --dataset_split_name=train-human \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${model_name} \
  --checkpoint_path=${TRAIN_DIR}/${four_stage_dir} \
  --max_number_of_steps=${train_seven_steps} \
  --batch_size=${batch_seven_size} \
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
  --train_length=7 \
  --num_clones=${num_clones} \
  --use_decoder=True

python eval_sgmt.py \
  --checkpoint_path=${TRAIN_DIR}/${seven_stage_dir} \
  --eval_dir=${TRAIN_DIR}/${second_stage_dir} \
  --dataset_name=${dataset_init_name} \
  --dataset_split_name=val-human-2016 \
  --dataset_dir=${DATASET_INIT_DIR} \
  --model_name=${model_name} \
  --batch_size=8 \
  --min_resize_value=512 \
  --max_resize_value=512 \
  --train_list=False \
  --use_decoder=True \
  --max_number_of_evaluations 1  # 0 for infinite loop


python train_video_sgmt.py \
  --train_dir=${TRAIN_DIR}/${ten_stage_dir} \
  --dataset_name=${dataset_name} \
  --dataset_split_name=train-human \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${model_name} \
  --checkpoint_path=${TRAIN_DIR}/${seven_stage_dir} \
  --max_number_of_steps=${train_ten_steps} \
  --batch_size=${batch_ten_size} \
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
  --train_length=10 \
  --num_clones=${num_clones} \
  --use_decoder=True

python eval_sgmt.py \
  --checkpoint_path=${TRAIN_DIR}/${ten_stage_dir} \
  --eval_dir=${TRAIN_DIR}/${second_stage_dir} \
  --dataset_name=${dataset_init_name} \
  --dataset_split_name=val-human-2016 \
  --dataset_dir=${DATASET_INIT_DIR} \
  --model_name=${model_name} \
  --batch_size=8 \
  --min_resize_value=512 \
  --max_resize_value=512 \
  --train_list=False \
  --use_decoder=True \
  --max_number_of_evaluations 1  # 0 for infinite loop



