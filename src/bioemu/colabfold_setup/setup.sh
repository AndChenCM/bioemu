#!/bin/bash

set -ex

echo "Setting up colabfold..."
BASE_PYTHON=$1
VENV_FOLDER=$2

export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

${BASE_PYTHON} -m venv --without-pip ${VENV_FOLDER}
${BASE_PYTHON} -m uv pip install --python ${VENV_FOLDER}/bin/python 'colabfold[alphafold-minus-jax]==1.5.4' 
${BASE_PYTHON} -m uv pip install --python ${VENV_FOLDER}/bin/python --force-reinstall \
    "jax[cuda12]==0.4.35" \
    "numpy==1.26.4" \
    "nvidia-cublas-cu12==12.1.*" \
    "nvidia-cuda-cupti-cu12==12.1.*" \
    "nvidia-cuda-nvcc-cu12==12.1.*" \
    "nvidia-cuda-runtime-cu12==12.1.*" \
    "nvidia-cusparse-cu12==12.1.*" \
    "nvidia-nvjitlink-cu12==12.1.*" \
    "nvidia-cudnn-cu12==9.3.*" \
    "nvidia-nccl-cu12" \
    "nvidia-cufft-cu12" \
    "nvidia-cusolver-cu12"

# Patch colabfold install
echo "Patching colabfold installation..."
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SITE_PACKAGES_DIR=${VENV_FOLDER}/lib/python3.*/site-packages
patch ${SITE_PACKAGES_DIR}/alphafold/model/modules.py ${SCRIPT_DIR}/modules.patch
patch ${SITE_PACKAGES_DIR}/colabfold/batch.py ${SCRIPT_DIR}/batch.patch

touch ${VENV_FOLDER}/.COLABFOLD_PATCHED
echo "Colabfold installation complete!"