#!/bin/bash
###
# @Author: DouZSh 6712525+douzsh@users.noreply.github.com
# @Date: 2025-05-12 09:01:26
# @FilePath: scripts/train.bash
# @Description: Script for training, evaluating and exporting models
###

set -e  # Exit immediately if a command exits with non-zero status

# ---- Constants ----
readonly CURRENT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
readonly ROOT_DIR=$(cd "${CURRENT_DIR}/.." && pwd)

# ---- Default Parameters ----
STAGE="train"                     # Operation mode: train/eval/export
LR="0.0005"                       # Learning rate
DEPTH_LIST="6,6,12,12"            # Model depth configuration
MEMBERS=8                         # Members for evaluation
VERBOSE_MODE=0                    # Verbose logging flag
SAVE_MODE=0                       # Save member averages flag
NUM_GPUS=1                        # Number of GPUs to use
BATCH_PER_GPU=1                   # Batch size per GPU
MASTER_ADDR=127.0.0.1             # Master address for distributed training
NNODES=1                          # Total number of nodes
SLEEP_TIME=0                     # Sleep time before starting (in seconds)
additional_args=()                # Additional arguments to pass to Python script
CONFIG_FILE="${ROOT_DIR}/configs/s2s_vit.py"
TRAIN_SCRIPT="${ROOT_DIR}/tools/train.py"

# ---- Functions ----
function print_usage() {
    cat << EOF
Usage: $(basename "$0") [options]

Options:
  --stage, -s STAGE         Specify the stage to run (train, ltrain, eval, leval, export). Default: '${STAGE}'
  --model, -m CONFIG        Path to model config file. Default: '${CONFIG_FILE}'
  --lr, -l RATE             Specify the learning rate. Default: '${LR}'
  --depth-list, -d DEPTHS   Model depth configuration as comma-separated values. Default: '${DEPTH_LIST}'
  --members NUM             Set number of members for evaluation. Default: ${MEMBERS}
  --save                    Save the member average results to a file
  --gpu, -g NUM             Specify the number of GPUs to use. Default: ${NUM_GPUS}
  --batch, -b SIZE          Specify the batch size per GPU. Default: ${BATCH_PER_GPU}
  --master-addr, -a ADDR    Specify the master address for distributed training. Default: ${MASTER_ADDR}
  --nnodes, -n NUM          Specify the total number of nodes. Default: ${NNODES}
  --sleep TIME              Sleep for TIME seconds before starting. Default: ${SLEEP_TIME}
  --verbose, -v             Enable verbose logging mode
  --help, -h                Show this help message and exit
EOF
}

function setup_environment() {
    # Create temporary config file
    CONFIG_FILE_TMP="${ROOT_DIR}/configs/tmp_config.py"
    cp "${CONFIG_FILE}" "${CONFIG_FILE_TMP}"
    
    # Format depth list for directory naming
    export DEPTH_LIST="${DEPTH_LIST}"
    export LR="${LR}"
    local temp_list=$(echo "${DEPTH_LIST}" | sed 's/,/_/g')  # replace , with _ in depth_list
    
    # Extract config file name without path and extension
    local config_file_name=$(basename "${CONFIG_FILE}" .py)
    
    # Set output directories
    export OUTPUT_DIR="${ROOT_DIR}/output/${temp_list}-${config_file_name}-${LR}/"
    
    # Set data directory and base config based on model type
    data_linux="/workspace/data/"
    data_mac="/Users/douzsh/Desktop/workspace/data/"
    # check os mac or linux
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "Running on macOS"
        data_dir="$data_mac"
        additional_args+=("train.device_type='mps'")
        additional_args+=("train.use_ddp=False")
        additional_args+=("train.use_fsdp=False")
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "Running on Linux"
        data_dir="$data_linux"
    else
        echo "Unsupported OS: $OSTYPE"
        exit 1
    fi

    if [[ "$config_file_name" == *"s2s"* ]]; then # for s2s model tests
        if [[ "$config_file_name" == *"cra"* ]]; then
            export DATA_DIR="${data_dir}/cra.1979_2024.c87.p25.h24/"
        elif [[ "$config_file_name" == *"p25"* ]]; then
            export DATA_DIR="${data_dir}/era5.1959_2023.c87.p25.h24/"
        else
            export DATA_DIR="${data_dir}/s2s.1950-2024.c76/"
        fi
        # check if string s2s_base exist in ${CONFIG_FILE_TMP}
        if grep -q "s2s_base" "${CONFIG_FILE_TMP}"; then
            export BASE_CONFIG="${ROOT_DIR}/configs/s2s_base.py"
        else
            export BASE_CONFIG="${CONFIG_FILE_TMP}"
        fi
    elif [[ "$config_file_name" == *"land"* ]]; then # for s2s model with land
        # Data dir is a dataset array, separated by colon
        export DATA_DIR="${data_dir}/s2s.1950-2024.c76:${data_dir}/land.1950-2024.c8/"
        if grep -q "land_base" "${CONFIG_FILE_TMP}"; then
            export BASE_CONFIG="${ROOT_DIR}/configs/land_base.py"
        else
            export BASE_CONFIG="${CONFIG_FILE_TMP}"
        fi
    elif [[ "$config_file_name" == *"ens"* || "$config_file_name" == *"iwc"* ]]; then
        if [[ "$config_file_name" == *"cra"* ]]; then
            export DATA_DIR="${data_dir}/cra.1979-2025.c71.p25.h6/"
        else
            export DATA_DIR="${data_dir}/era5.1979-2024.c78.p25.h6/"
        fi
        if grep -q "iwc_base" "${CONFIG_FILE_TMP}"; then
            export BASE_CONFIG="${ROOT_DIR}/configs/iwc_base.py"
        else
            export BASE_CONFIG="${CONFIG_FILE_TMP}"
        fi
    else
        echo "Error: Unsupported model type in config file name: ${config_file_name}"
        exit 1
    fi
    
    # Set logging level based on verbose mode
    if [[ $VERBOSE_MODE -eq 1 ]]; then
        sed -i "s/logging.INFO/logging.DEBUG/g" "${CONFIG_FILE_TMP}"
        sed -i "s/logging.INFO/logging.DEBUG/g" "${BASE_CONFIG}"
    else
        sed -i "s/logging.DEBUG/logging.INFO/g" "${CONFIG_FILE_TMP}"
        sed -i "s/logging.DEBUG/logging.INFO/g" "${BASE_CONFIG}"
    fi
    
    # Handle save mode
    if [[ $SAVE_MODE -eq 1 ]]; then
        echo "Saving member average results to ${OUTPUT_DIR}"
    else
        # Remove save_dir from config file if it exists
        sed -i "/save_dir/d" "${CONFIG_FILE_TMP}"
        echo "Disable saving results."
    fi
    
    # Update batch size based on GPU count
    echo "Update total_batch_size to $((NNODES * BATCH_PER_GPU * NUM_GPUS)) with ${BASE_CONFIG}."
    sed -i "s/^total_batch_size.*/total_batch_size = $((NNODES * BATCH_PER_GPU * NUM_GPUS))/g" "${BASE_CONFIG}"

    # add --eval to additional args first if stage is eval or leval
    if [[ "$STAGE" == "eval" || "$STAGE" == "leval" ]]; then
        additional_args=("--eval" "${additional_args[@]}")
        sed -i "s/^members = .*/members = ${MEMBERS}/g" "${CONFIG_FILE_TMP}"
    fi
    # use train_lightning.py for ltrain and leval
    if [[ "$STAGE" == "ltrain" || "$STAGE" == "leval" ]]; then
        TRAIN_SCRIPT="${ROOT_DIR}/tools/train_lightning.py"
    fi
}

# ---- Parse command line arguments ----
while [[ $# -gt 0 ]]; do
    case $1 in
        --stage|-s)
            STAGE="$2"
            shift 2
            ;;
        --model|-m)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --lr|-l)
            LR="$2"
            shift 2
            ;;
        --depth-list|-d)
            DEPTH_LIST="$2"
            shift 2
            ;;
        --members)
            MEMBERS="$2"
            shift 2
            ;;
        --help|-h)
            print_usage
            exit 0
            ;;
        --gpu|-g)
            NUM_GPUS="$2"
            shift 2
            ;;
        --batch|-b)
            BATCH_PER_GPU="$2"
            shift 2
            ;;
        --save)
            SAVE_MODE=1
            shift
            ;;
        --master-addr|-a)
            MASTER_ADDR="$2"
            shift 2
            ;;
        --nnodes|-n)
            NNODES="$2"
            shift 2
            ;;
        --sleep)
            SLEEP_TIME="$2"
            shift 2
            ;;
        --verbose|-v)
            VERBOSE_MODE=1
            shift
            ;;
        *)
            additional_args+=("$1")
            shift
            ;;
    esac
done

# ---- Validate inputs ----
echo "Stage: $STAGE with additional args: ${additional_args[*]}"

if [[ "$STAGE" != "train" && "$STAGE" != "eval" && "$STAGE" != "export" && "$STAGE" != "ltrain" && "$STAGE" != "leval" ]]; then
    echo "Error: Invalid stage: $STAGE"
    print_usage
    exit 1
fi

# ---- Run command after sleep ----
if [[ "${SLEEP_TIME}" -gt 0 ]]; then
    echo "Sleeping for ${SLEEP_TIME} seconds before starting..."
    sleep "${SLEEP_TIME}"
fi

# if NNODES == 1, set NODE_RANK to 0
if [[ "${NNODES}" -eq 1 ]]; then
    export NODE_RANK=0
    export LOCAL_ADDR=${MASTER_ADDR}
else
    export LOCAL_ADDR=$(hostname -I | awk '{print $1}')
    export NODE_RANK=$(( $(hostname -I | awk '{print $1}' | awk -F. '{print $4}') - 1 ))
fi
export HOSTNAME=node${NODE_RANK}
export MASTER_PORT=29500           # 推荐固定

# NCCL 设置
#export NCCL_DEBUG=INFO
#export NCCL_SOCKET_IFNAME=net

# ---- Setup environment ----
setup_environment

# ---- Display configuration ----
echo "Configuration:"
echo "  Using ${NUM_GPUS} GPU(s)"
echo "  Depth List:       ${DEPTH_LIST}"
echo "  Learning Rate:    ${LR}"
echo "  Data Directory:   ${DATA_DIR}"
echo "  Output Directory: ${OUTPUT_DIR}"
echo "  Master ADDR:      ${MASTER_ADDR}"
echo "  Local ADDR:       ${LOCAL_ADDR}"
echo "  Total Nodes:      ${NNODES}"
echo "  Node Rank:        ${NODE_RANK}"
echo "  HostName:         ${HOSTNAME}"

# ---- Run the specified stage ----
case $STAGE in
    train|eval|ltrain|leval)
        echo "Starting ${STAGE}ing..."
        sed -i "s/^export_onnx.*/export_onnx = False/g" "${BASE_CONFIG}"
        torchrun --nproc_per_node "${NUM_GPUS}" \
                 --node_rank "${NODE_RANK}" \
                 --nnodes "${NNODES}" \
                 --master_addr "${MASTER_ADDR}" \
                 --local-addr "${LOCAL_ADDR}" \
                 --master-port "${MASTER_PORT}" \
                 ${TRAIN_SCRIPT} \
                 --config-file "${CONFIG_FILE_TMP}" \
                 --num-gpus "${NUM_GPUS}" \
                 --num-machines "${NNODES}" \
                 "${additional_args[@]}"
        ;;
    export)
        echo "Starting export..."
        export CUDA_VISIBLE_DEVICES=0
        sed -i "s/^export_onnx.*/export_onnx = True/g" "${BASE_CONFIG}"
        python "${ROOT_DIR}/tools/train.py" \
               --config-file "${CONFIG_FILE_TMP}" \
               --export \
               "${additional_args[@]}"
        ;;
esac