#!/bin/bash
#PBS -q normal
#PBS -j oe
#PBS -l select=1:ncpus=32:ngpus=2:mem=220gb 
#PBS -l walltime=20:00:00
#PBS -P personal-sehwan00
#PBS -N log_train_ddpm_ema
# export NCCL_SOCKET_IFNAME=enp8s0
export NCCL_DEBUG=INFO
# Commands start here
module load pytorch/1.11.0-py3-gpu
cd ${PBS_O_WORKDIR}
torchrun --nproc_per_node=2 ./train_ddpm_ema.py