#!/usr/bin/env bash

set -x

PARTITION=$1
JOB_NAME=$2
CONFIG=$3
NODES=$4
GPUS_PER_NODE=4
CPUS_PER_TASK=40
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:5}
GPUS=`expr $GPUS_PER_NODE \* $NODES`

mkdir log

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
MASTER_PORT=${MASTER_PORT:-`python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()'`}
nohup srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    -N ${NODES} \
    -t 24:00:00 \
    ${SRUN_ARGS} \
    python -u $(dirname "$0")/train.py ${CONFIG} --launcher="slurm" ${PY_ARGS} &> log/${JOB_NAME}.log &

