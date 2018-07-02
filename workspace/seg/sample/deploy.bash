#!/bin/bash

dataset_name="coco-instance-simple"
tfrecord_dir="coco-instance/tfrecord_simple"
log_dir="coco-instance/model_instance_reshape_2/all"

model_name="mobilenet_v2_035_instance_2"
ckpt="${log_dir}/model.ckpt-400000"

###############################################
HOME="/home/corp.owlii.com/xiufeng.huang"
SLIM="${HOME}/models/research/slim"
TFBAZEL="${HOME}/tensorflow/bazel-bin"
WORKSPACE="${HOME}/models/workspace/seg"
TF="${HOME}/tensorflow/tensorflow"
DATASET_DIR="${WORKSPACE}/${tfrecord_dir}"
INIT_CHECKPOINT="${WORKSPACE}/${ckpt}"
DEPLOY_DIR="${WORKSPACE}/${log_dir}/deploy"
mkdir -p ${DEPLOY_DIR}

##############################################
extra_step=1
pbtxt="${DEPLOY_DIR}/graph.pbtxt"
ckpt="${DEPLOY_DIR}/model.ckpt-${extra_step}"
frozen="${DEPLOY_DIR}/frozen_graph.pb"
output_node="MobilenetV2/heatmap:0"
#output_node="ResizeBilinear_1:0"
#output_node="MobilenetV2/Logits/output:0"

input_node="image:0"
deploy="${DEPLOY_DIR}/deploy_graph.pb"

pbtxt_name="deploy_graph.pbtxt"

deploy_mlmodel="${DEPLOY_DIR}/deploy_graph.mlmodel"
ml_output_node=None  # use None when the logits is already 512x512
#ml_output_node="MobilenetV2/ResizeBilinear:0"
#ml_output_node="MobilenetV2/heatmap:0"
#input_size=[1,512,512,3]


# Prepare the graph for freezing
python train_sgmt.py \
  --train_dir=${DEPLOY_DIR} \
  --dataset_name=${dataset_name} \
  --dataset_split_name=coco_train_instance \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${model_name} \
  --checkpoint_path=${INIT_CHECKPOINT} \
  --max_number_of_steps=${extra_step} \
  --batch_size=1 \
  --learning_rate=0.0 \
  --learning_rate_decay_type=fixed \
  --weight_decay=0.00004 \
  --use_decoder=True \
  --adjust_for_deploy=true 


echo ${ckpt}
# Freeze the graph
python ${WORKSPACE}/my_graph_utils/freeze_graph.py \
    --input_graph=${pbtxt} \
    --input_binary=false \
    --input_checkpoint=${ckpt} \
    --output_graph=${frozen} \
    --output_node_names=${output_node::-2} \

${TFBAZEL}/tensorflow/tools/graph_transforms/transform_graph \
    --in_graph=${frozen} \
    --out_graph=${deploy} \
    --inputs=${input_node} \
    --outputs=${output_node} \
    --transforms='strip_unused_nodes(type=float, shape="1,512,512,3")
                  remove_nodes(op=Identity, op=CheckNumerics)
                  fold_constants(ignore_errors=true)
                  fold_batch_norms
                  fold_old_batch_norms'
#                  add_default_attributes'

python ${WORKSPACE}/my_graph_utils/pb2pbtxt.py \
    --pb_path=${deploy} \
    --log_dir=${DEPLOY_DIR} \
    --pbtxt_name=${pbtxt_name}


. /home/corp.owlii.com/yi.xu/anaconda2/etc/profile.d/conda.sh
python ${WORKSPACE}/tf_coreml_utils/tf2coreml.py \
    --input_pb_file=${deploy} \
    --output_mlmodel=${deploy_mlmodel} \
    --input_node_name=${input_node} \
    --output_node_name=${ml_output_node}


result_dir="${DEPLOY_DIR}/deployed_graph"
mkdir -p "${result_dir}"
cp "${deploy_mlmodel}" "${result_dir}/"
cp "${deploy}" "${result_dir}/"
cp "${DEPLOY_DIR}/${pbtxt_name}" "${result_dir}/"


