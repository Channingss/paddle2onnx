cmake_minimum_required(VERSION 3.0)
project(PaddleX CXX C)

SET(TRITON_DIR "" CACHE PATH "Location of libraries")
SET(GLOG_DIR "" CACHE PATH "Location of libraries")
SET(GFLAGS_DIR "" CACHE PATH "Location of libraries")
SET(OPENCV_DIR "" CACHE PATH "Location of libraries")

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -o2  -lcurl -fopenmp -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -o2  -fopenmp -std=c++11")
#set(CMAKE_STATIC_LIBRARY_PREFIX "")

if (NOT WIN32)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/demo)
else()
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/paddlex_inference)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/paddlex_inference)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/paddlex_inference)
endif()

if (NOT WIN32)
    SET(YAML_BUILD_TYPE ON CACHE BOOL "yaml build shared library.")
else()
    SET(YAML_BUILD_TYPE OFF CACHE BOOL "yaml build shared library.")
endif()
include_directories("${CMAKE_SOURCE_DIR}/")

include(cmake/yaml-cpp.cmake)
include_directories("${CMAKE_CURRENT_BINARY_DIR}/ext/yaml-cpp/src/ext-yaml-cpp/include")
link_directories("${CMAKE_CURRENT_BINARY_DIR}/ext/yaml-cpp/lib")

MESSAGE("${GFLAGS_DIR}")
include_directories("${GFLAGS_DIR}/include")
link_directories("${GFLAGS_DIR}/lib")

include_directories("${GLOG_DIR}/include")
link_directories("${GLOG_DIR}/lib")

include_directories("${TRITON_DIR}/include")
link_directories("${TRITON_DIR}/lib")


find_package(OpenCV REQUIRED PATHS ${OPENCV_DIR}/share/OpenCV NO_DEFAULT_PATH)
#find_package(TRITON  REQUIRED PATHS ${TRITON_DIR}/lib/cmake/TRITON NO_DEFAULT_PATH)

macro(safe_set_static_flag)
    foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
      if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      endif(${flag_var} MATCHES "/MD")
    endforeach(flag_var)
endmacro()

#set(DEPS ${TRITON_DIR}/lib/libhttpclient_static${CMAKE_STATIC_LIBRARY_SUFFIX})
#set(DEPS ${TRITON_DIR}/lib/libcurl${CMAKE_STATIC_LIBRARY_SUFFIX})
#set(DEPS ${TRITON_DIR}/lib/libaddress_sorting${CMAKE_STATIC_LIBRARY_SUFFIX})
#set(DEPS ${TRITON_DIR}/lib/libgpr${CMAKE_STATIC_LIBRARY_SUFFIX})
#set(DEPS ${TRITON_DIR}/lib/libcares${CMAKE_STATIC_LIBRARY_SUFFIX})
#set(DEPS ${TRITON_DIR}/lib/libtritontableprinter${CMAKE_STATIC_LIBRARY_SUFFIX})
#set(DEPS ${TRITON_DIR}/lib/libprotobuf${CMAKE_STATIC_LIBRARY_SUFFIX})
set(DEPS ${TRITON_DIR}/lib/libhttpclient${CMAKE_SHARED_LIBRARY_SUFFIX})
set(DEPS ${DEPS} ${OpenCV_LIBS})
#set(DEPS ${DEPS} ${TRITON_LIBS})
set(DEPS ${DEPS} glog gflags protobuf yaml-cpp)

if (NOT WIN32)
    set(EXTERNAL_LIB "-ldl -lrt -lgomp -lz -lm -lpthread")
    set(DEPS ${DEPS} ${EXTERNAL_LIB})
endif()

#add_library(paddlex_inference SHARED src/engine/triton_engine.cpp src/preprocess/ppdet_pre_proc.cpp src/common/config.cpp)
#ADD_DEPENDENCIES(paddlex_inference ext-yaml-cpp)
#target_link_libraries(paddlex_inference ${DEPS})

add_executable(ppdet_infer demo/triton_inference/ppdet_infer.cpp src/engine/triton_engine.cpp src/postprocess/ppdet_post_proc.cpp src/preprocess/ppdet_pre_proc.cpp src/preprocess/preprocess.cpp src/common/config.cpp src/common/transforms.cpp)
ADD_DEPENDENCIES(ppdet_infer ext-yaml-cpp)
target_link_libraries(ppdet_infer ${DEPS})
