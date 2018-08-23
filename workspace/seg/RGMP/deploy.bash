#!/bin/bash


project="sample"
log_dir="RGMP/modelRGMP1/ten"
HOME="/home/corp.owlii.com/xiufeng.huang"
#HOME="/home/corp.owlii.com/yi.xu"
TFBAZEL="${HOME}/tensorflow/bazel-bin"
ROOT="${HOME}/models/workspace/seg"
WORKSPACE="${ROOT}/${project}"
PYTHONPATH="${PYTHONPATH}:${WORKSPACE}:${WORKSPACE}/.."
DEPLOY_DIR="${ROOT}/${log_dir}/deploy"
mkdir -p ${DEPLOY_DIR}

ckpt="${ROOT}/${log_dir}/model.ckpt-200000"
model_name="mobilenet_v2_1_video_trainseq"
frozen_name="frozen_graph"
deploy_name="deploy_graph"
input_node="image:0,first:0"
output_node="heatmap:0"
input_shape="1,512,512,4"
output_size="512,512"
num_classes=2


python deploy.py \
    --trained_checkpoint_prefix=${ckpt} \
    --output_directory=${DEPLOY_DIR} \
    --output_graph_name="${frozen_name}.pb" \
    --input_shape=${input_shape} \
    --output_size=${output_size} \
    --num_classes=${num_classes} \
    --model_name=${model_name} \
    --train_list=False \
    --use_decoder=True \
    --deploy_output_resizer=False

${TFBAZEL}/tensorflow/tools/graph_transforms/transform_graph \
    --in_graph="${DEPLOY_DIR}/${frozen_name}.pb" \
    --out_graph="${DEPLOY_DIR}/${deploy_name}.pb" \
    --inputs=${input_node} \
    --outputs=${output_node} \
    --transforms='strip_unused_nodes
                  remove_nodes(op=Identity, op=CheckNumerics)
                  fold_constants(ignore_errors=true)
                  fold_batch_norms
                  fold_old_batch_norms'

python ${ROOT}/my_graph_utils/pb2pbtxt.py \
    --pb_path="${DEPLOY_DIR}/${deploy_name}.pb" \
    --log_dir=${DEPLOY_DIR} \
    --pbtxt_name="${deploy_name}.pbtxt"

python ${ROOT}/tf_coreml_utils/tf2coreml.py \
    --input_pb_file="${DEPLOY_DIR}/${deploy_name}.pb" \
    --output_mlmodel="${DEPLOY_DIR}/${deploy_name}.mlmodel" \
    --input_node_name=${input_node} \
    --output_node_name=${output_node}

# turn the mlmodel output type of MLMultiArray to Image (PixelBuffer in iOS)
mv "${DEPLOY_DIR}/${deploy_name}.mlmodel" "${DEPLOY_DIR}/mlma_${deploy_name}.mlmodel"
ml_output_node="heatmap__0"
python ${ROOT}/tf_coreml_utils/multiarray2image.py \
    --input_mlmodel="${DEPLOY_DIR}/mlma_${deploy_name}.mlmodel" \
    --output_mlmodel="${DEPLOY_DIR}/${deploy_name}.mlmodel" \
    --output_node=${ml_output_node} \
    --is_bgr=False
