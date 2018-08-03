#!/bin/bash


################
# Each image and its mask must have the same prefix name and same relative position to their folder so that they can be described by one list file, though their location folder and file format suffix can be different.
###############


#DATA_ROOT="/OwlNAS/OwlDocuments/OwlStudio/data/sgmt-data"
#WORK_DIR="/home/corp.owlii.com/yi.xu/data/owlii_studio"

DATA_ROOT="/home/corp.owlii.com/yi.xu/data/pascal_voc_seg/VOCdevkit/VOC2012/saliency"
OUTPUT_DIR="${DATA_ROOT}/tfrecord"
mkdir -p "${OUTPUT_DIR}"

num_shards=5
IMAGE_FOLDER="${DATA_ROOT}/../JPEGImages"
MASK_FOLDER="${DATA_ROOT}/mask"
LIST_FOLDER="${DATA_ROOT}/list"
python ./img2tf.py \
  --image_folder="${IMAGE_FOLDER}" \
  --semantic_segmentation_folder="${MASK_FOLDER}" \
  --list_folder="${LIST_FOLDER}" \
  --image_format="jpg" \
  --label_format="png" \
  --num_shards=${num_shards} \
  --output_dir="${OUTPUT_DIR}"



