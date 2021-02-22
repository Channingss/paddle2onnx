TRITON_DIR=/workspace/install/
WITH_STATIC_LIB=OFF
#{
#    bash $(pwd)/scripts/bootstrap.sh # 下载预编译版本的加密工具和opencv依赖库
#} || {
#    echo "Fail to execute script/bootstrap.sh"
#    exit -1
#}

OPENCV_DIR=$(pwd)/deps/opencv3.4.6gcc4.8ffmpeg/
GLOG_DIR=$(pwd)/deps/glog/
GFLAGS_DIR=$(pwd)/deps/gflags/

rm -rf build
mkdir -p build
cd build
cmake .. \
    -DTRITON_DIR=${TRITON_DIR} \
    -DWITH_STATIC_LIB=${WITH_STATIC_LIB} \
    -DOPENCV_DIR=${OPENCV_DIR}  \
    -DGLOG_DIR=${GLOG_DIR} \
    -DGFLAGS_DIR=${GFLAGS_DIR}

make -j16
