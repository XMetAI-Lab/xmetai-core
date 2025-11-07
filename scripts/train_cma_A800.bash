#!/usr/bin/bash

readonly CURRENT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
readonly ROOT_DIR=$(cd "${CURRENT_DIR}/.." && pwd)

# env config
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=OFF
export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
export NCCL_IB_HCA=mlx5_0,mlx5_1
export NCCL_IB_MERGE=1  # 可选，默认是 1
#export NCCL_IB_GID_INDEX=3

# check ib info
#ibv_devinfo
#cat /sys/class/infiniband/mlx5_0/ports/1/gid_attrs/ndevs/*
#nvidia-smi topo -m

# disable ib for tempera
#export NCCL_IB_DISABLE=1

# start dataset preparation
echo "------------------------------------"
echo "check data from s2s" 
DATA_DIR="/gpu/COMMONDATA/CMME-S2S/ERA5/s2s.1950_2023.c76"
ls -l ${DATA_DIR}
echo "------------------------------------"


# update config batch size
CONFIG_FILE="${ROOT_DIR}/configs/s2s_fix.py"
BASE_CONFIG="${ROOT_DIR}/configs/s2s_base.py"
# update total_batch_size to ${NNODES} * 8
if [ -f "${CONFIG_FILE}" ]; then
    echo "Updated total_batch_size in ${BASE_CONFIG} to $((NNODES * 8))"
    sed -i "s/total_batch_size = 1/total_batch_size = $((NNODES * 8))/g" "${BASE_CONFIG}"
else
    echo "Config file ${BASE_CONFIG} not found!"
    exit 1
fi

# set model configs
export DEPTH_LIST="12,12"
export LR="0.005"
temp_list=$(echo ${DEPTH_LIST} | sed 's/,/_/g')  # replace , to _ in depth_list
# get config_file name without path and extension
config_file_name=$(basename "${CONFIG_FILE}" .py)
export OUTPUT_DIR="/gpu/douzsh/s2s/output-${temp_list}-${config_file_name}-${LR}/"
export DATA_DIR="/gpu/COMMONDATA/CMME-S2S/ERA5/s2s.1950_2023.c76/s2s.1950_2023.c76/"

echo "Using depth_list: ${DEPTH_LIST}"
echo "Using learning rate: ${LR}"
echo "Using output directory: ${OUTPUT_DIR}"


# start training
torchrun --nproc_per_node ${NPROC:=1} \
         --node_rank ${NODE_RANK:=0} \
         --nnodes=${NNODES:=1} \
         --master_addr ${MASTER_ADDR:=127.0.0.1} \
         ${ROOT_DIR}/tools/train.py \
           --config-file ${CONFIG_FILE} \
           --num-machines ${NNODES:=1} --num-gpus 8 \
           --machine-rank ${NODE_RANK:=0} \
           --dist-url tcp://${MASTER_ADDR:=127.0.0.1}:23456