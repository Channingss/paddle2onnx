#!/bin/bash

for i in "$@"; do
    case $i in
        --triton_client=*)
         TRITON_CLIENT="${i#*=}"
         shift
         ;;
        *)
         echo "unknown option $i"
         exit 1
         ;;
    esac
done

if [ $TRITON_CLIENT ];then
	echo "TRITON_CLIENT = $TRITON_CLIENT"
else
	echo "TRITON_CLIENT is not exist, please set by --triton_client"
    exit 1
fi

# install relayed library
sh $(pwd)/scripts/triton_env.sh 

# download opencv library
OPENCV_DIR=$(pwd)/deps/opencv3.4.6gcc4.8ffmpeg/
{
    bash $(pwd)/scripts/bootstrap.sh ${OPENCV_DIR}
} || {
    echo "Fail to execute script/bootstrap.sh"
    exit -1
}

# download glog library
GLOG_DIR=$(pwd)/deps/glog/
GLOG_URL=https://bj.bcebos.com/paddlex/deploy/glog.tar.gz

if [ ! -d $(pwd)deps/ ]; then
    mkdir -p deps
fi

if [ ! -d ${GLOG_DIR} ]; then
    cd deps
    wget -c ${GLOG_URL} -O glog.tar.gz
    tar -zxvf glog.tar.gz 
    rm -rf glog.tar.gz 
    cd ..
fi

# download gflogs library
GFLAGS_DIR=$(pwd)/deps/gflags/
GFLAGS_URL=https://bj.bcebos.com/paddlex/deploy/gflags.tar.gz
if [ ! -d ${GFLAGS_DIR} ]; then
    cd deps
    wget -c ${GFLAGS_URL} -O glog.tar.gz
    tar -zxvf glog.tar.gz 
    rm -rf glog.tar.gz 
    cd ..
fi

rm -rf build
mkdir -p build
cd build
cmake ../demo/triton_inference/ \
    -DTRITON_CLIENT=${TRITON_CLIENT} \
    -DOPENCV_DIR=${OPENCV_DIR}  \
    -DGLOG_DIR=${GLOG_DIR} \
    -DGFLAGS_DIR=${GFLAGS_DIR}

make -j16
