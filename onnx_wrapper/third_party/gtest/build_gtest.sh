#!/bin/bash
### download `wget -c -O googletest-1.15.2.zip  https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip` in current path.

function find_project_root_path() {
    # echo "@i@ --> find dir: ${0}"
    this_script_dir=$( dirname -- "$0"; )
    pwd_dir=$( pwd; )
    if [ "${this_script_dir:0:1}" = "/" ]
    then
        # echo "get absolute path ${this_script_dir}" > /dev/tty
        project_root_path=${this_script_dir}"/"
    else
        # echo "get relative path ${this_script_dir}" > /dev/tty
        project_root_path=${pwd_dir}"/"${this_script_dir}"/"
    fi
    echo "${project_root_path}"
}

PRJ_ROOT_PATH=$( find_project_root_path )
echo "project_root_path: ${PRJ_ROOT_PATH}" 
cd ${PRJ_ROOT_PATH}

GTEST_SRC_PATH=googletest-1.15.2
CPU_ARCHITECTURE=`uname -m`
PORTING_COMPILE_PREFIX=${CPU_ARCHITECTURE}-linux-gnu
COMPILE_PREFIX=${PORTING_COMPILE_PREFIX}-
PROJECT_RUNTIME_PATH=${PRJ_ROOT_PATH}/output

rm -rf ${GTEST_SRC_PATH}
unzip ${GTEST_SRC_PATH}.zip
mkdir -p ${PROJECT_RUNTIME_PATH}
mkdir -p ${GTEST_SRC_PATH}/build
cd ${GTEST_SRC_PATH}/build
cmake .. \
-DCMAKE_BUILD_TYPE=Release \
-DBUILD_SHARED_LIBS=ON  \
-DBUILD_STATIC_LIBS=OFF \
-DCMAKE_C_COMPILER=${COMPILE_PREFIX}gcc \
-DCMAKE_CXX_COMPILER=${COMPILE_PREFIX}g++  \
-DCMAKE_SYSTEM_PROCESSOR=${PORTING_COMPILE_PREFIX} \
-DCMAKE_CXX_STANDARD=17 \
-DCMAKE_C_FLAGS="-I${PROJECT_RUNTIME_PATH}/include -fPIC -O3" \
-DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link=${PROJECT_RUNTIME_PATH}/lib" \
-DCMAKE_INSTALL_PREFIX=${PROJECT_RUNTIME_PATH}  \
-DCMAKE_VERBOSE_MAKEFILE=ON
make -j8 V=1
make install