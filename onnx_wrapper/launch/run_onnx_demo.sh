#!/bin/bash

function find_project_root_path() {
    # echo "@i@ --> find dir: ${0}"
    this_script_dir=$( dirname -- "$0"; )
    pwd_dir=$( pwd; )
    if [ "${this_script_dir:0:1}" = "/" ]
    then
        # echo "get absolute path ${this_script_dir}" > /dev/tty
        project_root_path=${this_script_dir}"/../../"
    else
        # echo "get relative path ${this_script_dir}" > /dev/tty
        project_root_path=${pwd_dir}"/"${this_script_dir}"/../../"
    fi
    echo "${project_root_path}"
}

PRJ_ROOT_PATH=$( find_project_root_path )
echo "project_root_path: ${PRJ_ROOT_PATH}" 
PRJ_SUB_PATH=${PRJ_ROOT_PATH}/onnx_wrapper
cd ${PRJ_SUB_PATH}

CPU_ARCHITECTURE=`uname -m`
USE_ORT_VERSION=1.16.3
ORT_LIB_PATH=${PRJ_SUB_PATH}/third_party/onnxruntime/${USE_ORT_VERSION}/lib/${CPU_ARCHITECTURE}
GTEST_LIB_PATH=${PRJ_SUB_PATH}/third_party/gtest/lib/${CPU_ARCHITECTURE}

export LD_LIBRARY_PATH=${GTEST_LIB_PATH}:${ORT_LIB_PATH}:$LD_LIBRARY_PATH
echo "CPU_ARCHITECTURE: ${CPU_ARCHITECTURE}, LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"

ONNX_MODEL_PATH=${PRJ_SUB_PATH}/model/pointpillars_mmdet3d_nus.onnx
# ONNX_MODEL_PATH=${PRJ_ROOT_PATH}/model/pointpillar.onnx

install/onnx_wrapper_demo/bin/onnx_wrapper_demo_bin  ${ONNX_MODEL_PATH}