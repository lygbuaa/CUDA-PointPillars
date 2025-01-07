#!/bin/bash

### follow https://mmdeploy.readthedocs.io/en/latest/04-supported-codebases/mmdet3d.html
### 3rd-Party Prerequisites:
# python3 -m pip install -U openmim
# python3 -m mim install "mmdet3d>=1.1.0"
# pip install mmdeploy

MMDEPLOY_ROOT_PATH=github/open_mmlab/mmdeploy
POINTPILLARS_MODEL_PATH=pointpillars
cd ${MMDEPLOY_ROOT_PATH}
mkdir -p ${POINTPILLARS_MODEL_PATH}
mim download mmdet3d --config pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d --dest ${POINTPILLARS_MODEL_PATH}

export MODEL_CONFIG=${POINTPILLARS_MODEL_PATH}/pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d.py
export MODEL_PATH=${POINTPILLARS_MODEL_PATH}/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20210826_225857-f19d00a3.pth
export TEST_DATA=tests/data/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151612397179.pcd.bin

python3 tools/deploy.py configs/mmdet3d/voxel-detection/voxel-detection_onnxruntime_dynamic.py $MODEL_CONFIG $MODEL_PATH $TEST_DATA --work-dir ${POINTPILLARS_MODEL_PATH}
